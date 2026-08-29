import torch

def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.

    Parameters
    ----------
    pts_src
        [R, D] matrix
    pts_dst
        C, D] matrix
    p
        p-norm
    
    Return
    ------
    [R, C] matrix
        distance matrix
    """
    return torch.cdist(pts_src, pts_dst, p)  # Euclidean distance

def unbalanced_ot(tran, latent_A, latent_B, device, Couple, reg=0.1, reg_m=1.0):
    '''
    Calculate a unbalanced optimal transport matrix between batches.

    Parameters
    ----------
    tran
        transport matrix between the two batches sampling from the global OT matrix. 
    mu1
        mean vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    reg
        Entropy regularization parameter in OT. Default: 0.1
    reg_m
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    '''
    ns = latent_A.size(0)  # number of samples in modality A
    nt = latent_B.size(0)  # number of samples in modality B
    
    cost_pp = distance_matrix(latent_A, latent_B)
    cost_pp = cost_pp.to(device) 

    if Couple is not None:
        Couple = torch.tensor(Couple, dtype=torch.float).to(device)

    p_s = torch.ones(ns, 1) / ns
    #else:

    p_t = torch.ones(nt, 1) / nt
    #else:

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if tran is None:
        tran = torch.ones(ns, nt,) / (ns * nt)
        tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f = reg_m / (reg_m + reg)

    for m in range(10):
        if Couple is not None:
            cost = cost_pp*Couple
        else:
            cost = cost_pp

        denominator = reg * torch.max(torch.abs(cost))
        if denominator.item() == 0:  # avoid division by zero
            kernel = tran
        else:
            kernel = torch.exp(-cost /denominator) * tran
        
        kernel = torch.nan_to_num(kernel, nan=1e-8, posinf=1e8, neginf=-1e8)
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(10):
            dual =( p_s / (kernel @ b) )**f
            b = ( p_t / (torch.t(kernel) @ dual) )**f
            dual = torch.nan_to_num(dual, nan=1e-8, posinf=1e8, neginf=-1e8)
            b = torch.nan_to_num(b, nan=1e-8, posinf=1e8, neginf=-1e8)

        tran = (dual @ torch.t(b)) * kernel
        tran = torch.nan_to_num(tran, nan=1e-8, posinf=1e8, neginf=-1e8)

    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    uot_loss  = (cost * tran.detach().data).sum()

    return uot_loss , tran.detach()


def unbalanced_ot_parameter(tran, latent_A, latent_B, device, Couple, reg=0.1, reg_m_1 = 1, reg_m_2 = 1):
    '''
    Calculate a unbalanced optimal transport matrix between batches with different reg_m parameters.

    Parameters
    ----------
    tran
        transport matrix between the two batches sampling from the global OT matrix. 
    mu1
        mean vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    reg
        Entropy regularization parameter in OT. Default: 0.1
    reg_m_1
        Unbalanced OT parameter 1. Larger values means more balanced OT. Default: 1.0
    reg_m_2
        Unbalanced OT parameter 2. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    '''

    ns = latent_A.size(0)  # number of samples in modality A
    nt = latent_B.size(0)  # number of samples in modality B

    cost_pp = distance_matrix(latent_A, latent_B)
    cost_pp = cost_pp.to(device) 

    if Couple is not None:
        Couple = torch.tensor(Couple, dtype=torch.float).to(device)

    p_s = torch.ones(ns, 1) / ns
    p_t = torch.ones(nt, 1) / nt

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if tran is None:
        tran = torch.ones(ns, nt) / (ns * nt)
        tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)

    f1 = reg_m_1 / (reg_m_1 + reg)
    f2 = reg_m_2 / (reg_m_2 + reg)

    for m in range(10):
        if Couple is not None:
            cost = cost_pp*Couple
        else:
            cost = cost_pp

        denominator = reg * torch.max(torch.abs(cost))
        if denominator.item() == 0:  # avoid division by zero
            kernel = tran
        else:
            kernel = torch.exp(-cost /denominator) * tran
        
        kernel = torch.nan_to_num(kernel, nan=1e-8, posinf=1e8, neginf=-1e8)
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(10):
            dual =( p_s / (kernel @ b) )**f
            b = ( p_t / (torch.t(kernel) @ dual) )**f
            dual = torch.nan_to_num(dual, nan=1e-8, posinf=1e8, neginf=-1e8)
            b = torch.nan_to_num(b, nan=1e-8, posinf=1e8, neginf=-1e8)

        tran = (dual @ torch.t(b)) * kernel
        tran = torch.nan_to_num(tran, nan=1e-8, posinf=1e8, neginf=-1e8)

    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    uot_loss  = (cost * tran.detach().data).sum()

    return uot_loss,tran.detach()