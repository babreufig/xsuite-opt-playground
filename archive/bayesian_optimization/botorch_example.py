import typing as t

import torch

from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf

import matplotlib.pyplot as plt


def objective_fun(X: torch.tensor) -> torch.tensor:
    # 2D parabola
    x1, x2 = X[..., 0], X[..., 1]
    return -(x1**2 + 3.0 * x2**2)


def plot_evaluation(
    gp_model: SingleTaskGP,
    train_x: torch.tensor,
    train_Y: torch.tensor,
    objective_fun: t.Callable[[torch.tensor], torch.tensor],
    param_bounds: torch.tensor,
    n_eval_x1: int = 100,
    n_eval_x2: int = 90,
    vmin: float = -20.0,
    vmax: float = 0.0,
) -> None:
    """Plot ground truth and predicted mean and stddev."""

    # Evaluate objective on grid
    x1 = torch.linspace(param_bounds[0, 0], param_bounds[1, 0], n_eval_x1)
    x2 = torch.linspace(param_bounds[0, 1], param_bounds[1, 1], n_eval_x2)
    x1_mg, x2_mg = torch.meshgrid(x1, x2, indexing="ij")
    grid_X = torch.stack([x1_mg, x2_mg], dim=-1)
    grid_y = objective_fun(grid_X)

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
) -> SingleTaskGP:
    """Build GP model with integrated scalers for in- and output."""
    outcome_transform = Standardize(m=1)
    outcome_transform.train()

    n_params = param_bounds.size(1)
    gp_model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=n_params, bounds=param_bounds),
        outcome_transform=outcome_transform,
    )
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)
    return gp_model


def get_next_candidate(
    gp_model: SingleTaskGP,
    param_bounds: torch.tensor,
    beta: float = 2.0,
) -> torch.tensor:
    """Optimize acquisition function to get next candidate."""
    acqf = UpperConfidenceBound(gp_model, beta=beta)
    candidate, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=param_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
    )
    return candidate


def get_predictions(gp_model: SingleTaskGP, X: torch.tensor) -> torch.tensor:
    """Get predictions from GP model (mean and stddev)."""
    with torch.no_grad():
        posterior = gp_model.posterior(X)
        pred_mean = posterior.mean
        pred_stddev = posterior.variance.sqrt()
    return pred_mean, pred_stddev


N_INITIAL = 7
N_POINTS_BO = 10
PARAM_BOUNDS = torch.tensor([[-4.0, -5.0], [4.0, 5.0]], dtype=torch.float64)


# I) Collect initial random points (could be also just 1 point to start)
train_X = PARAM_BOUNDS[0] + (PARAM_BOUNDS[1] - PARAM_BOUNDS[0]) * torch.rand(
    N_INITIAL,
    PARAM_BOUNDS.size(1),
    dtype=torch.float64,
)
# Simulation
train_Y = objective_fun(train_X).unsqueeze(-1)

# II) BO loop
for it in range(N_POINTS_BO):
    # a) Initialize and fit model with latest data
    gp_model = build_gp_model(train_X, train_Y, PARAM_BOUNDS)

    # b) Optimize acquisition function to get next candidate
    candidate = get_next_candidate(gp_model, PARAM_BOUNDS, beta=2.0)
    print(f"{it=}: {candidate=}")

    # c) Evaluate objective function (e.g. on machine, in simulation,
    # etc.) and update data sets
    new_y = objective_fun(candidate).unsqueeze(-1)
    train_X = torch.cat([train_X, candidate], dim=0)
    train_Y = torch.cat([train_Y, new_y], dim=0)

    # d) (optional) Plot evaluation
    plot_evaluation(gp_model, train_X, train_Y, objective_fun, PARAM_BOUNDS)