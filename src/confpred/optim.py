from copy import deepcopy
from typing import Callable, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn
import tqdm


def unconstrained_opt(
    loss_func: Callable[[Tensor, Tensor], Tensor],
    dataloader: DataLoader,
    params: List[nn.Parameter],
    lr: float = 2e-4,
    max_iter: int = 500,
    term_threshold: float = 1e-4,
    term_loss_smoothing: float = 0.95,
    log_fn: Optional[Callable[[dict], None]] = None,
    optim_state_dict: Optional[dict] = None,
    pos_params: Optional[List[nn.Parameter]] = None,
):
    if pos_params is not None:
        params = params + pos_params

    # set up opimizer
    optim = torch.optim.Adam(params=params, lr=lr)
    if optim_state_dict:
        optim.load_state_dict(optim_state_dict)

    # instatiate data
    data_iter = iter(dataloader)

    t = tqdm.trange(max_iter)
    old_avg_loss = None
    avg_loss = None
    losses = []
    converged = False
    for i in t:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(params[0].device)
        targets = targets.to(params[0].device)

        l = loss_func(inputs, targets)
        l.backward()

        optim.step()

        # project pos_params
        if pos_params:
            for p in pos_params:
                p.data.clamp_(min=0)

        optim.zero_grad()

        curr_loss = l.item()

        if avg_loss is None:
            avg_loss = l
        else:
            avg_loss = (
                1 - term_loss_smoothing
            ) * curr_loss + term_loss_smoothing * avg_loss

        losses.append(avg_loss)

        if i > 0 and i % 50 == 0:
            # TODO: use gradient magnitude as termination criterion
            # log avg_loss and check for termination
            if old_avg_loss is not None:
                if torch.abs(old_avg_loss - avg_loss) < term_threshold:
                    # improvement too small, terminating
                    converged = True
                    break
            # otherwise, log average loss at this time
            old_avg_loss = avg_loss

        t.set_postfix_str(f"{curr_loss=:0.04f},{avg_loss=:0.04f}")

        if log_fn:
            log_fn(
                {
                    "epoch": i,
                    "curr_loss": curr_loss,
                    "avg_loss": avg_loss,
                }
            )

    return losses, converged, optim.state_dict()


def avg_loss_constraint(loss_and_constraint_func, dataloader, device):
    losses = []
    constraints = []
    for (inputs, targets) in tqdm.tqdm(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            loss, constraint = loss_and_constraint_func(inputs, targets)
            losses.append(loss.detach().cpu().numpy())
            constraints.append(constraint.detach().cpu().numpy())

    loss = np.mean(losses)
    constraint = np.mean(constraints)

    return loss, constraint


def iterative_penalty_opt(
    loss_and_constraint_func: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
    dataloader: DataLoader,
    params: List[nn.Parameter],
    max_outer_iter: int = 10,
    lr=2e-4,
    lr_decay=1.,
    **kwargs,
):
    """
    Iteratively penalize KL until constraint is met. TODO: implement this

    Args:
        loss_and_constraint_func (Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]): _description_
        dataloader (DataLoader): _description_
        params (List[nn.Parameter]): _description_
        max_outer_iter (int, optional): _description_. Defaults to 10.
        lr (_type_, optional): _description_. Defaults to 2e-4.
        lr_decay (_type_, optional): _description_. Defaults to 1..
    """
    params = list(params)
    # store old params
    old_param_vals = [p.data.clone() for p in params]
    

    lam = 0
    loss, constraint = avg_loss_constraint(
        loss_and_constraint_func, dataloader, params[0].device
    )
    init_constraint = constraint
    print(f"Upon initialization: {loss=}, {constraint=}")
    
    # store best params
    best_loss = loss
    best_constraint = constraint
    best_param_vals = old_param_vals

    
    if constraint > 0:
        print("No feasible solution (initialization wasn't feasible)")
        return False, best_loss, best_constraint
    
    min_lam = 0
    max_lam = 1
    lam = 1e-2
    optim_state_dict = None
    for i in range(max_outer_iter):
        print(f"Trying {lam=}") 
        def lagrangian(inputs, targets):
            loss, constraint = loss_and_constraint_func(inputs, targets)
            penalty = constraint - init_constraint
            return loss + lam*penalty
        
        unconstrained_opt(
            lagrangian,
            dataloader,
            params,
            lr=lr,
            optim_state_dict=optim_state_dict,
            **kwargs,
        )

        # Evaluate avg loss and constraint
        loss, constraint = avg_loss_constraint(
            loss_and_constraint_func, dataloader, params[0].device
        )

        print(f"-> {loss=}, {constraint=}")
        
        if constraint <= 0:  # and loss < best_loss:
            print(f"-> Feasible!")
            # if improvement, update best_loss
            if loss <= best_loss:
                best_loss = loss
                best_constraint = constraint
                best_param_vals = [p.data.clone() for p in params]
            
            # midpoint search to lower lam
            max_lam = lam            
            lam = (lam + min_lam) / 2
        else:
            # midpoint search to higher lam
            min_lam = lam
            lam = (lam + max_lam) / 2

            
        # reset param values for next try
        for p, old_val in zip(params, old_param_vals):
            p.data = old_val

    feasible_found = best_constraint < 0

    for p, best_val in zip(params, best_param_vals):
        p.data = best_val

    return feasible_found, best_loss, best_constraint

def aug_lag_constrained_opt(
    loss_and_constraint_func: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
    dataloader: DataLoader,
    params: List[nn.Parameter],
    max_outer_iter: int = 10,
    lr=2e-4,
    lr_decay=1.,
    **kwargs,
) -> Tuple[bool, float, float]:
    """
    Solves inequality constrained problem using augmented lagragian with the penalty method

    Args:
        model (nn.Module): _description_
        loss_func (Callable[[], Tensor]): _description_
        constraint_func (Callable[[], Tensor]): _description_
        params (List[nn.Parameter]): _description_
        max_outer_iters (int, optional): _description_. Defaults to 10.

    Returns:
        Tuple[bool, float, float]: _description_
    """
    params = list(params)
    # store old params
    old_param_vals = [p.data.clone() for p in params]

    # initialize penalty scale
    lam = 0.0  # multiplier
    rho = 1e-3  #
    growth_rate = np.exp(np.log(3./rho)/max_outer_iter)
    slack = nn.Parameter(torch.zeros(1, device=params[0].device))
    s = slack.item()

    loss, constraint = avg_loss_constraint(
            loss_and_constraint_func, dataloader, params[0].device
        )
    print(f"Upon initialization: {loss=}, {constraint=}, {s=}")

    # store best params
    best_loss = loss
    best_constraint = constraint
    best_param_vals = old_param_vals

    
    if constraint > 0:
        print("No feasible solution (initialization wasn't feasible)")
        return False, best_loss, best_constraint

    optim_state_dict = None
    for i in range(max_outer_iter):
        # define penalized objective

        def augmented_lagragian(inputs, targets):
            loss, constraint = loss_and_constraint_func(inputs, targets)
            equal_constraint = constraint + slack.mean() #torch.relu(constraint)
            loss += lam * equal_constraint
            loss += rho / 2 * equal_constraint**2
            return loss

        # optimize
        print(f"Trying {lam=}, {rho=}, {lr=}")
        _, _, optim_state_dict = unconstrained_opt(
            augmented_lagragian,
            dataloader,
            params,
            lr=lr,
            optim_state_dict=optim_state_dict,
            pos_params=[slack],
            **kwargs,
        )

        # Evaluate avg loss and constraint
        loss, constraint = avg_loss_constraint(
            loss_and_constraint_func, dataloader, params[0].device
        )
        s = slack.item()

        print(f"-> {loss=}, {constraint=}, {s=}")

        lam += rho * (constraint + s) #max(0,constraint)
        rho = min(3.0, growth_rate * rho)
        
        lr = max(lr_decay*lr, 1e-8)

        if constraint <= 0:  # and loss < best_loss:
            print(f"-> Feasible! {loss=}, {constraint=}, {s=}")
            # if improvement, update best_loss
            if loss <= best_loss:
                best_loss = loss
                best_constraint = constraint
                best_param_vals = [p.data.clone() for p in params]

    feasible_found = best_loss < np.inf

    for p, best_val in zip(params, best_param_vals):
        p.data = best_val

    return feasible_found, best_loss, best_constraint
