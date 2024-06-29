import optuna
import joblib

"""Script to read the saved pickle file after hyperparameter tuning"""

def load_study(study_path):
    """Loads the Optuna study object from file."""
    return joblib.load(study_path)

def analyze_study(study):
    """Analyzes the Optuna study object to print statistics and the best trial."""
    completed_trials = study.trials_dataframe()

    # Print overall statistics
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)}")
    print(f"  Number of complete trials: {sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)}")

    # Best trial
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Optionally, you can also plot optimization results using Optuna's visualization module
    # For example, plot the optimization history
    try:
        from optuna.visualization import plot_optimization_history
        plot_optimization_history(study)
    except ImportError:
        print("Optuna visualization not available.")

if __name__ == "__main__":
    study_path = '/home/prsh7458/work/R2D/hyperparameter/Intense/study.pkl'
    study = load_study(study_path)
    analyze_study(study)
