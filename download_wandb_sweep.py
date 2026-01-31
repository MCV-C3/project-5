import wandb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast

# wandb config
ENTITY = "project-5"
PROJECT = "C3-Week1-BOVW"

OUTPUT_PATH = "/home/bernat/MCV/C3/project/project-5/Week1/wandb_csv/"

# SWEEP ID to filter runs (don't delete the previous sweeps)
SWEEP_ID = "pr0w0qnp" # Sweep with codebook test from 4 to 2048 crossval 

# next sweep IDs to export:
# SWEEP_ID = "..."

# Define a function to delete a single run
def export_run(run):
    try:
        # Collect run's summary metrics, configs, and name
        summary = run.summary._json_dict
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        name = run.name

        # Combine summary and config into a single dictionary
        run_data = {**summary, **config}
        run_data['name'] = name

        run_data['id'] = run.id
        run_data['created_at'] = run.created_at
        run_data['state'] = run.state
        try:
            run_data['sweep_name'] = run.sweep_name
        except:
            run_data['sweep_name'] = None
        return run_data

    except Exception as e:
        return f"Error exporting run {run.id}: {e}"


api = wandb.Api()
runs = api.runs(f'{ENTITY}/{PROJECT}')

runs_data = []

with ThreadPoolExecutor(max_workers=9) as executor:
    for run_data in executor.map(export_run, runs):
        if run_data["sweep_name"] == SWEEP_ID:
            runs_data.append(run_data)

runs_df = pd.DataFrame(runs_data)

cols = ['id', 'name', 'created_at', 'state'] + [col for col in runs_df.columns if col not in ['id', 'name', 'created_at', 'state']]
runs_df = runs_df[cols]
# runs_df.to_csv(f"/home/bernat/MCV/C3/project/project-5/Week1/wandb_csv/runs_{SWEEP_ID}.csv", index=False)
columns_to_save = ['id','name','_runtime','_step','_timestamp','test','train','folds_data',
                 'n_pca','scale','stride','use_pca','pyramid_lvls','codebook_size',
                 'detector_type','detector_kwargs','classifier_kwargs',
                 'classifier_algorithm','normalize_histograms','sweep_name']


# df_refined = pd.read_csv(f"/home/bernat/MCV/C3/project/project-5/Week1/wandb_csv/runs_{SWEEP_ID}.csv")[columns_to_save]
df_refined = runs_df[columns_to_save]

# Convert 'test' and 'train' columns from string to dictionary
print(df_refined['test'].head())
df_refined['test'] = df_refined['test'].astype(str)
df_refined['train'] = df_refined['train'].astype(str)
df_refined['folds'] = df_refined['folds'].astype(str)
df_refined['test_metrics'] = df_refined['test'].apply(ast.literal_eval)
df_refined['train_metrics'] = df_refined['train'].apply(ast.literal_eval)
df_refined['folds_data'] = df_refined['folds'].apply(ast.literal_eval)


# Extract key scalar metrics into new columns
def extract_scalar_metrics(df, source_col, prefix):
    for key in df[source_col].iloc[0].keys():
        if 'per_class' in key:
            for class_key in df[source_col].iloc[0][key].keys():
                df[f'{prefix}_{key}_{class_key}'] = df[source_col].apply(lambda x: x.get(key).get(class_key))
        else:
            df[f'{prefix}_{key}'] = df[source_col].apply(lambda x: x.get(key))
        
    
    # df[f'{prefix}_accuracy'] = df[source_col].apply(lambda x: x.get('accuracy'))
    # df[f'{prefix}_AUC_macro'] = df[source_col].apply(lambda x: x.get('AUC_macro'))
    # df[f'{prefix}_f1_macro'] = df[source_col].apply(lambda x: x.get('f1_macro'))
    return df

df_refined = extract_scalar_metrics(df_refined, 'test_metrics', 'test')
df_refined = extract_scalar_metrics(df_refined, 'train_metrics', 'train')
df_refined = extract_scalar_metrics(df_refined, 'folds_data', 'folds')

# Export DataFrame to CSV
df_refined.to_csv(f"{OUTPUT_PATH}/metrics_{SWEEP_ID}.csv", index=False)

print(f"Data has been successfully exported to '{OUTPUT_PATH}/metrics_{SWEEP_ID}.csv'.")