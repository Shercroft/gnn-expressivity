"""Run GraphGPS with the shared BREC CLI safeguards."""

from reproduce_gin_brec import main as run_brec_cli
from gnn_expressivity.training.brec_evaluation import evaluate_graphgps_brec


def main() -> None:
    run_brec_cli(model_name="graphgps", evaluator=evaluate_graphgps_brec)


if __name__ == "__main__":
    main()
