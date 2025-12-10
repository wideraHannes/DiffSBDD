import torch
import torch.nn as nn
import torch.nn.functional as F
from equivariant_diffusion.egnn_new import EGNN, GNN
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion

remove_mean_batch = EnVariationalDiffusion.remove_mean_batch
import numpy as np


class EGNNDynamics(nn.Module):
    def __init__(
        self,
        atom_nf,
        residue_nf,
        n_dims,
        joint_nf=16,
        hidden_nf=64,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
        mode="egnn_dynamics",
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
        update_pocket_coords=True,
        edge_cutoff_ligand=None,
        edge_cutoff_pocket=None,
        edge_cutoff_interaction=None,
        reflection_equivariant=True,
        edge_embedding_dim=None,
        use_film=True,
        pca_lambda=4.0,
    ):
        super().__init__()
        self.mode = mode
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf), act_fn, nn.Linear(2 * atom_nf, joint_nf)
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf), act_fn, nn.Linear(2 * atom_nf, atom_nf)
        )
        # das würde ich durch das embedding aus ESM-C ersetzen residue nf = 960
        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, joint_nf),
        )

        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf),
        )

        self.edge_embedding = (
            nn.Embedding(3, self.edge_nf) if self.edge_nf is not None else None
        )
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        # ============================================================================
        # COMMENTED OUT: ESM-C FiLM conditioning network (replaced with PCA approach)
        # ============================================================================
        # self.use_film = use_film
        # self.film_network = nn.Sequential(
        #     nn.Linear(960, 2 * hidden_nf),
        #     act_fn,
        #     nn.Linear(2 * hidden_nf, hidden_nf),
        #     act_fn,
        #     nn.Linear(hidden_nf, joint_nf),
        # )
        # self.film_lambda = nn.Parameter(torch.tensor(0.01))

        # ============================================================================
        # PCA-based ESM-C pocket embedding approach
        # ============================================================================
        # Instead of learning projection via FiLM network, we use pre-trained PCA
        # to project ESM-C embeddings from 960D to joint_nf dimensions
        #
        # Formula: h_residues = z_pocket + λ * z_esm_pca
        # where:
        #   z_pocket: encoded residue features (joint_nf dimensional)
        #   z_esm_pca: PCA-projected ESM-C embeddings (960D -> joint_nf)
        #   λ: scaling factor (configurable, default 4.0, set to 1.0 to disable scaling)
        self.use_pca = use_film  # Reuse flag for PCA approach
        self.pca_model = None  # Will be loaded from pickle file
        self.pca_lambda = pca_lambda  # Configurable scaling factor for PCA contribution

        if condition_time:
            dynamics_node_nf = joint_nf + 1
        else:
            print("Warning: dynamics model is _not_ conditioned on time.")
            dynamics_node_nf = joint_nf

        if mode == "egnn_dynamics":
            self.egnn = EGNN(
                in_node_nf=dynamics_node_nf,
                in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                reflection_equiv=reflection_equivariant,
            )
            self.node_nf = dynamics_node_nf
            self.update_pocket_coords = update_pocket_coords

        elif mode == "gnn_dynamics":
            self.gnn = GNN(
                in_node_nf=dynamics_node_nf + n_dims,
                in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf,
                out_node_nf=n_dims + dynamics_node_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )

        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time

    def forward(
        self, xh_atoms, xh_residues, t, mask_atoms, mask_residues, pocket_emb=None
    ):
        x_atoms = xh_atoms[:, : self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims :].clone()

        x_residues = xh_residues[:, : self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims :].clone()

        # embed atom features and residue features in a shared space
        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(
            h_residues
        )  # z_pocket: [num_residues, joint_nf]

        # ============================================================================
        # COMMENTED OUT: FiLM-based ESM-C conditioning
        # ============================================================================
        # if self.use_film and pocket_emb is not None:
        #     delta = self.film_network(pocket_emb)  # [batch_size, joint_nf]
        #     delta_expanded = delta[mask_residues.long()]  # [num_residues, joint_nf]
        #     h_residues = h_residues + self.film_lambda * delta_expanded
        #     print(f"FiLM λ: {self.film_lambda.item():.4f}, delta mean: {delta.mean().item():.4f}")

        # ============================================================================
        # PCA-based ESM-C pocket embedding
        # ============================================================================
        # Formula: h_residues = z_pocket + λ * z_esm_pca
        if self.use_pca and pocket_emb is not None and self.pca_model is not None:
            # pocket_emb shape: [batch_size, 960] (ESM-C embeddings)

            # Project ESM-C embeddings using PCA: 960D -> joint_nf
            # Note: PCA model should be trained to project to joint_nf dimensions
            z_esm_pca = self.pca_model.transform(
                pocket_emb.cpu().numpy()
            )  # [batch_size, joint_nf]
            z_esm_pca = torch.from_numpy(z_esm_pca).float().to(pocket_emb.device)

            # Expand to per-residue using mask_residues
            z_esm_pca_expanded = z_esm_pca[
                mask_residues.long()
            ]  # [num_residues, joint_nf]

            # Add scaled PCA embeddings to encoded residue features
            h_residues = h_residues + self.pca_lambda * z_esm_pca_expanded

            # Monitor PCA contribution (optional debug info)
            print(
                f"PCA λ: {self.pca_lambda:.4f}, "
                f"z_esm_pca mean: {z_esm_pca_expanded.mean().item():.4f}, "
                f"z_esm_pca std: {z_esm_pca_expanded.std().item():.4f}"
            )

        # combine the two node types AFTER FiLM is applied to residues
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[mask]
            # $1 just concatinate h with the pocket encoding
            h = torch.cat([h, h_time], dim=1)

        # theroetisch sollte jetzt die dimension von h
        # anstatt h_residues einfach esmc rein packen
        # # AA_seq_Length x (h_atoms + h_residues (hier viel platz für ideen) + time)

        # für eine Aminosäure in einem pocket haben wir
        # x: 3 dimensional
        # h: 16 dimnesional (welches atom)
        # idee-h*: 960 dimensional (protein embedding)

        # get edges of a complete graph
        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)
        assert torch.all(mask[edges[0]] == mask[edges[1]])

        # Get edge types
        if self.edge_nf > 0:
            # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket
            edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
            edge_types[(edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))] = 1
            edge_types[
                (edges[0] >= len(mask_atoms)) & (edges[1] >= len(mask_atoms))
            ] = 2

            # Learnable embedding
            edge_types = self.edge_embedding(edge_types)
        else:
            edge_types = None

        if self.mode == "egnn_dynamics":
            update_coords_mask = (
                None
                if self.update_pocket_coords
                else torch.cat(
                    (torch.ones_like(mask_atoms), torch.zeros_like(mask_residues))
                ).unsqueeze(1)
            )
            h_final, x_final = self.egnn(
                h,
                x,
                edges,
                update_coords_mask=update_coords_mask,
                batch_mask=mask,
                edge_attr=edge_types,
            )
            vel = x_final - x

        elif self.mode == "gnn_dynamics":
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=None, edge_attr=edge_types)
            vel = output[:, :3]
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        # decode atom and residue features
        h_final_atoms = self.atom_decoder(h_final[: len(mask_atoms)])
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms) :])

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
            else:
                raise ValueError("NaN detected in EGNN output")

        if self.update_pocket_coords:
            # in case of unconditional joint distribution, include this as in
            # the original code
            vel = remove_mean_batch(vel, mask)

        return torch.cat([vel[: len(mask_atoms)], h_final_atoms], dim=-1), torch.cat(
            [vel[len(mask_atoms) :], h_final_residues], dim=-1
        )

    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (
                torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l
            )

        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (
                torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p
            )

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (
                torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i
            )

        adj = torch.cat(
            (
                torch.cat((adj_ligand, adj_cross), dim=1),
                torch.cat((adj_cross.T, adj_pocket), dim=1),
            ),
            dim=0,
        )
        edges = torch.stack(torch.where(adj), dim=0)

        return edges
