import torch
import torch.nn.functional as F

def compute_laminate_stiffness(plies, materials):
    """
    Compute equivalent ABD matrix for a laminate using Classical Laminate Theory.
    Returns A, B, D matrices (3x3 each) as torch tensors.
    
    Args:
        plies: Tensor of shape [num_plies, 3] with [angle, thickness, material_id]
        materials: Dict mapping material_id to material properties
    """
    A = torch.zeros(3, 3, device=plies.device)
    B = torch.zeros(3, 3, device=plies.device)
    D = torch.zeros(3, 3, device=plies.device)

    z_k = torch.zeros(len(plies) + 1, device=plies.device)
    for i in range(len(plies)):
        z_k[i + 1] = z_k[i] + plies[i, 1]  # thickness

    h_total = z_k[-1]

    for i in range(len(plies)):
        t_k = plies[i, 1]
        z_mid = (z_k[i] + z_k[i + 1]) / 2

        angle = plies[i, 0] * torch.pi / 180.0
        mat_id = int(plies[i, 2].item())
        mat = materials[mat_id]

        # Convert material properties to tensors
        E1 = torch.tensor(mat['EX'], dtype=torch.float32, device=plies.device)
        E2 = torch.tensor(mat['EY'], dtype=torch.float32, device=plies.device)
        G12 = torch.tensor(mat['GXY'], dtype=torch.float32, device=plies.device)
        nu12 = torch.tensor(mat['PRXY'], dtype=torch.float32, device=plies.device)
        
        nu21 = nu12 * E2 / E1

        Q11 = E1 / (1 - nu12 * nu21)
        Q22 = E2 / (1 - nu12 * nu21)
        Q12 = nu12 * E2 / (1 - nu12 * nu21)
        Q66 = G12

        c = torch.cos(angle)
        s = torch.sin(angle)
        c2 = c**2
        s2 = s**2
        c4 = c2**2
        s4 = s2**2
        cs2 = c2*s2

        # Compute transformed stiffness matrix components
        Q11_bar = Q11*c4 + 2*(Q12 + 2*Q66)*cs2 + Q22*s4
        Q12_bar = Q12*(s4 + c4) + (Q11 + Q22 - 4*Q66)*cs2
        Q16_bar = (Q11 - Q12 - 2*Q66)*c*s*(c2 - s2)
        Q22_bar = Q11*s4 + 2*(Q12 + 2*Q66)*cs2 + Q22*c4
        Q26_bar = (Q11 - Q12 - 2*Q66)*c*s*(s2 - c2)
        Q66_bar = Q66*(s4 + c4) + (Q11 + Q22 - 2*Q12 - 2*Q66)*cs2

        # Build Qbar matrix properly
        Qbar = torch.tensor([
            [Q11_bar.item(), Q12_bar.item(), Q16_bar.item()],
            [Q12_bar.item(), Q22_bar.item(), Q26_bar.item()],
            [Q16_bar.item(), Q26_bar.item(), Q66_bar.item()]
        ], dtype=torch.float32, device=plies.device)

        A += Qbar * t_k
        B += Qbar * t_k * z_mid
        D += Qbar * (t_k * z_mid**2 + t_k**3 / 12)

    return A / h_total, B / h_total, D / h_total, h_total


def virtual_work_loss(batch, pred_disp, pressure=-500.0, lambda_phys=1.0):
    """
    Compute |W_int - W_ext|² for a batch of graphs.
    Assumes:
    - Shell elements (membrane + bending)
    - Uniform pressure on top face (z > z_max - eps)
    - Mid-plane displacements predicted
    
    Args:
        batch: PyG Batch object with graph data
        pred_disp: Predicted displacements [total_nodes, 3]
        pressure: Applied pressure value (MPa)
        lambda_phys: Weighting factor for physics loss
    """
    device = pred_disp.device
    total_residual = 0.0
    batch_size = batch.num_graphs

    for graph_idx in range(batch_size):
        mask = batch.batch == graph_idx
        pos = batch.pos[mask]           # [N, 3]
        u_pred = pred_disp[mask]        # [N, 3]

        # Extract plies and materials for this graph
        try:
            # Handle both padded and unpadded formats
            if hasattr(batch, 'plies') and hasattr(batch, 'plies_mask'):
                # New format with padding
                plies_padded = batch.plies[graph_idx]  # [max_plies, 3]
                mask_plies = batch.plies_mask[graph_idx]  # [max_plies]
                plies = plies_padded[mask_plies]  # Filter to actual plies
            elif hasattr(batch, 'plies'):
                # Old format or list
                plies = batch.plies[graph_idx]
            else:
                # Fallback: skip physics loss for this graph
                continue

            # Get materials dictionary
            if hasattr(batch, 'materials'):
                if isinstance(batch.materials, list):
                    materials = batch.materials[graph_idx]
                else:
                    materials = batch.materials
            else:
                continue

            # Compute laminate stiffness
            A, B, D, h = compute_laminate_stiffness(plies, materials)
            print(A,B,C,h)
            
        except Exception as e:
            print(f"Warning: CLT failed for graph {graph_idx}: {e}")
            continue

        # Approximate strains from displacement field
        strain_mem = torch.zeros(pos.shape[0], 3, device=device)
        strain_curv = torch.zeros(pos.shape[0], 3, device=device)

        # Use neighbor averaging (like mean strain per node)
        edge_index = batch.edge_index
        for i in range(pos.shape[0]):
            # Get neighbors for this node in this graph
            global_i = torch.where(mask)[0][i]
            nbrs_mask = edge_index[0] == global_i
            nbrs_global = edge_index[1][nbrs_mask]
            
            # Convert back to local indices
            nbrs_local = []
            for n in nbrs_global:
                local_idx = torch.where(torch.where(mask)[0] == n)[0]
                if len(local_idx) > 0:
                    nbrs_local.append(local_idx[0].item())
            
            if len(nbrs_local) == 0:
                continue
                
            nbrs = torch.tensor(nbrs_local, device=device)
            dx = pos[nbrs] - pos[i:i+1]
            du = u_pred[nbrs] - u_pred[i:i+1]
            dist = torch.norm(dx[:, :2], dim=1, keepdim=True).clamp_min(1e-6)
            dx_norm = dx[:, :2] / dist

            # In-plane strains
            eps_x = (du[:, 0:1] * dx_norm[:, 0:1]).sum(dim=0) / (dist.mean() + 1e-8)
            eps_y = (du[:, 1:2] * dx_norm[:, 1:2]).sum(dim=0) / (dist.mean() + 1e-8)
            eps_xy = (du[:, 0:1] * dx_norm[:, 1:2] + du[:, 1:2] * dx_norm[:, 0:1]).sum(dim=0) / (dist.mean() + 1e-8)

            strain_mem[i] = torch.stack([eps_x.squeeze(), eps_y.squeeze(), eps_xy.squeeze()])

            # Very rough curvature estimate from out-of-plane displacement
            if len(nbrs) >= 3:
                # Simple finite difference approximation
                d2u_z = (u_pred[nbrs, 2].mean() - u_pred[i, 2]) / ((dist.mean() ** 2) + 1e-8)
                strain_curv[i, 2] = -d2u_z * h / 2  # approximate

        # Internal work: ε_mem : A : ε_mem + κ : D : κ  (per unit area)
        W_int_mem = 0.5 * torch.einsum('ni,ij,nj->n', strain_mem, A, strain_mem)
        W_int_bend = 0.5 * torch.einsum('ni,ij,nj->n', strain_curv, D, strain_curv)
        W_int = (W_int_mem + W_int_bend).mean() * h

        # External work: pressure × average uz on loaded surface
        z_max = pos[:, 2].max()
        loaded_nodes = pos[:, 2] > z_max - 1e-3
        if loaded_nodes.sum() > 0:
            avg_uz = u_pred[loaded_nodes, 2].mean()
            area_per_node = 1.0  # approximate
            W_ext = -pressure * avg_uz * loaded_nodes.sum() * area_per_node
        else:
            W_ext = torch.tensor(0.0, device=device)

        residual = W_int - W_ext
        total_residual += residual ** 2

    physics_loss = total_residual / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)
    return lambda_phys * physics_loss
