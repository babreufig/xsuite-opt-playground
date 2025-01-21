import xtrack as xt
import os
import numpy as np

import typing as t
import torch

from botorch.models import MultiTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from botorch.optim import optimize_acqf

import matplotlib.pyplot as plt


dir_path = os.path.dirname(os.path.realpath(__file__))

# Load a line and build a tracker
line = xt.Line.from_json(dir_path + '/lhc_thick_with_knobs.json')
line.build_tracker()

# Match tunes and chromaticities to assigned values
opt = line.match(
    solve=True,
    method='4d',
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8, limits=[-1e-4, 1e-4], tag='quad'),
        #xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),
    ],
    targets = [
        xt.TargetSet(qx=62.315, qy=60.325, tol=1e-6, tag='tune'),
        #xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),
    ])

merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)
opt.check_limits = False
bounds = merit_function.get_x_limits()


def objective_fun(X: torch.tensor) -> torch.tensor:
    # minus for maximization problem
    return -merit_function(X)


def get_ground_truth_2d(
    objective_fun: t.Callable[[torch.tensor], torch.tensor],
    param_bounds: torch.tensor,
    n_eval_x1: int = 10,
    n_eval_x2: int = 9,
):
    # Evaluate objective on grid
    x1 = torch.linspace(param_bounds[0, 0], param_bounds[1, 0], n_eval_x1)
    x2 = torch.linspace(param_bounds[0, 1], param_bounds[1, 1], n_eval_x2)
    x1_mg, x2_mg = torch.meshgrid(x1, x2, indexing="ij")
    grid_X = torch.stack([x1_mg, x2_mg], dim=-1)
    grid_y = np.zeros((n_eval_x1, n_eval_x2))
    for i in range(n_eval_x1):
        for j in range(n_eval_x2):
            grid_y[i, j] = objective_fun(grid_X[i, j])
    
    grid_y = np.array(grid_y)
    return x1, x2, grid_y

def plot_evaluation_2d(
    x1: torch.tensor,
    x2: torch.tensor,
    grid_y: torch.tensor,
    gp_model: MultiTaskGP,
    train_x: torch.tensor,
    train_Y: torch.tensor,
) -> None:
    """Plot ground truth and predicted mean and stddev."""
    vmin = np.min(grid_y)
    vmax = np.max(grid_y)
    x1_mg, x2_mg = torch.meshgrid(x1, x2, indexing="ij")
    grid_X = torch.stack([x1_mg, x2_mg], dim=-1)

    # Predictions
    pred_mean, pred_stddev = get_predictions(gp_model, grid_X)

    # Plot ground truth and predictions
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].set_title("Ground truth")
    axs[0].pcolormesh(x1_mg, x2_mg, grid_y, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")

    axs[1].set_title("Predicted mean")
    axs[1].pcolormesh(x1_mg, x2_mg, pred_mean.squeeze(), cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_xlabel("x1")
    axs[1].set_ylabel("x2")

    axs[2].set_title("Predicted stddev")
    axs[2].pcolormesh(x1_mg, x2_mg, pred_stddev.squeeze(), cmap="gray")
    axs[2].set_xlabel("x1")
    axs[2].set_ylabel("x2")

    # Add training points
    for ax in axs[:2]:
        ax.scatter(
            train_x[:-1, 0],
            train_x[:-1, 1],
            c=train_Y.squeeze()[:-1],
            linewidth=1,
            edgecolor="white",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.scatter(
            train_x[-1, 0],
            train_x[-1, 1],
            c=train_Y.squeeze()[-1],
            linewidth=1,
            edgecolor="red",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

    axs[2].scatter(
        train_x[:-1, 0],
        train_x[:-1, 1],
        linewidth=1,
        color="white",
        edgecolor="white",
    )
    axs[2].scatter(
        train_x[-1, 0],
        train_x[-1, 1],
        linewidth=1,
        color="red",
        edgecolor="red",
    )
    plt.tight_layout()
    plt.show()


def build_gp_model(
    train_X: torch.tensor,
    train_Y: torch.tensor,
    param_bounds: torch.tensor,
) -> MultiTaskGP:
    """Build GP model with integrated scalers for in- and output."""

    outcome_transform = Standardize(m=train_X.size(1))
    outcome_transform.train()

    n_params = param_bounds.size(1)

    likelihood = MultitaskGaussianLikelihood(num_tasks=train_Y.size(1))

    gp_model = MultiTaskGP(
        train_X=train_X, # (10, 4)
        train_Y=train_Y, # (10, 4)
        input_transform=Normalize(d=n_params, bounds=param_bounds),
        outcome_transform=outcome_transform,
        task_feature=-1,
        likelihood=likelihood
        #covar_module=MaternKernel(nu=2.5),
    )

    print(likelihood)

    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)
    return gp_model


def get_next_candidate(
    gp_model: MultiTaskGP,
    param_bounds: torch.tensor,
    beta: float
) -> torch.tensor:
    """Optimize acquisition function to get next candidate."""
    acqf = UpperConfidenceBound(gp_model, beta=beta)
    candidate, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=param_bounds,
        q=1,
        num_restarts=20,
        raw_samples=512,
    )
    return candidate


def get_predictions(gp_model: MultiTaskGP, X: torch.tensor) -> torch.tensor:
    """Get predictions from GP model (mean and stddev)."""
    with torch.no_grad():
        posterior = gp_model.posterior(X)
        pred_mean = posterior.mean
        pred_stddev = posterior.variance.sqrt()
    return pred_mean, pred_stddev


N_INITIAL = 10
PARAM_BOUNDS = torch.tensor(bounds.T, dtype=torch.float64)
TOL = 1e-8
EARLY_STOP_THRESHOLD = 10


# I) Collect initial random points (could be also just 1 point to start)
train_X = PARAM_BOUNDS[0] + (PARAM_BOUNDS[1] - PARAM_BOUNDS[0]) * torch.rand(
    N_INITIAL,
    PARAM_BOUNDS.size(1),
    dtype=torch.float64,
)
# Simulation
train_Y = []
for x in train_X:
    train_Y.append(objective_fun(x))
train_Y = torch.tensor(np.array(train_Y), dtype=torch.float64)#.unsqueeze(-1)

#x1, x2, grid_y = get_ground_truth_2d(objective_fun, PARAM_BOUNDS)

# II) BO loop
i = 0
early_stop_cnt = 0
best_y = np.inf

while True:
    print(f"{early_stop_cnt=}")
    # a) Initialize and fit model with latest data
    gp_model = build_gp_model(train_X, train_Y, PARAM_BOUNDS)

    beta = 2 * np.exp(-0.2 * i)

    # b) Optimize acquisition function to get next candidate
    candidate = get_next_candidate(gp_model, PARAM_BOUNDS, beta=beta)

    # c) Evaluate objective function (e.g. on machine, in simulation,
    # etc.) and update data sets
    new_y = torch.tensor(np.atleast_2d(objective_fun(candidate.flatten())), dtype=torch.float64)
    train_X = torch.cat([train_X, candidate], dim=0)
    train_Y = torch.cat([train_Y, new_y], dim=0)

    new_y = torch.sum(new_y**2)

    if torch.abs(new_y) < best_y:
        best_y = torch.abs(new_y)
        early_stop_cnt = 0
    else:
        print(new_y, best_y)
        early_stop_cnt += 1

    print(f"{i=}: {new_y=}, {candidate=}")

    # d) (optional) Plot evaluation
    # if i % 10 == 0:
    #     plot_evaluation_2d(x1, x2, grid_y, gp_model, train_X, train_Y)

    if torch.abs(new_y).item() < TOL:
        print("Converged")
        break

    if early_stop_cnt == EARLY_STOP_THRESHOLD:
        print("Early Stop Threshold reached")
        best_ind = torch.argmax(train_Y, dim=1)
        best_x = train_X[best_ind]
        print(f"Current solution: {best_x=}")
        break
    i += 1

if torch.abs(best_y).item() < TOL:
    print("Done")
else:
    print("Not done - needs further optimization...")
    merit_function.set_x(best_x)
    opt.target_status()
    print(f"Status: {i} Iterations, {merit_function.merit_function.call_counter} Calls, {best_y} penalty")
    x0 = best_x.numpy()
    opt.solve()
    #result = minimize(merit_function, x0=x0, bounds=bounds, method='L-BFGS-B')
    #print("Optimal solution:", result.x)
    #merit_function.set_x(result.x)
    print(f"Final number of calls: {merit_function.merit_function.call_counter} Calls\n")
opt.target_status()