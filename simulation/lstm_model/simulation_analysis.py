import os

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from joypy import joyplot

from pathlib import Path

from dotenv import load_dotenv


# def import_run_csv(project_name: str, save_path: str):
#     # it doesn't work! Fix the csv export from wandb
#     import wandb
#     import pandas as pd
#
#     # login to wandb
#     wandb.login()
#
#     # get the runs table summary
#     runs = wandb.Api().runs(project_name)
#
#     # create a list of columns to include in the CSV
#     # Select column of interest
#     columns_to_include = ['Name', 'batch_size', 'dropout', 'epochs', 'hidden_size', 'lb', 'learning_rate',
#                           'num_layer', 'optimizer_name', 'output_pred',
#                           'Loss Test', 'Loss Train', 'Loss Validation',
#                           'MAPE CL', 'MAPE Test', 'MAPE Train', 'MAPE Validation',
#                           'R2 CL', 'RMSE CL', 'RMSE Test', 'RMSE Train', 'RMSE Validation', 'epoch']
#
#     # create a list of dictionaries with the values for the specified columns
#     rows = []
#     for run in runs:
#         row = {}
#         for column in columns_to_include:
#             keys = column.split(".")
#             value = run
#             for key in keys:
#                 if isinstance(value, dict):
#                     value = value.get(key, "")
#                 else:
#                     value = ""
#             row[column] = value
#         rows.append(row)
#
#     # create a pandas dataframe with the rows
#     df = pd.DataFrame(rows)
#
#     # save the dataframe as a CSV file
#     df.to_csv(save_path, index=False)
#
#
# def selecet_runs_by_metrics(df: pd.DataFrame, save_dir: str, metric: str, threshold: float, file: str):
#     # Select column of interest
#     var = ['Name', 'architecture', 'batch_size', 'dropout', 'epochs',
#            'hidden_size', 'lb', 'learning_rate', 'num_layer', 'optimizer_name', 'Loss Test',
#            'Loss Train', 'Loss Validation', 'MAPE CL', 'MAPE Test', 'MAPE Train',
#            'MAPE Validation', 'R2 CL', 'RMSE CL', 'RMSE Test', 'RMSE Train',
#            'RMSE Validation', 'epoch']
#
#     # select clolum of df from var and select the rows that have a RMSE Test > 1
#     df = df[var][df[metric] > threshold]
#
#     # Extract the string after the last '_' and before the first '_' and insert it in a new column called id
#     df.insert(0, 'resstock_building_id', df['Name'].str.split('_').str[-1].str.split('_').str[0])
#
#     # add a column with the simulation reference at the first place
#     df.insert(0, 'file', file)
#
#     # Save df in a csv file
#     filename = save_dir + 'buildings_with_' + metric + f'_greater_{threshold}' + '.csv'
#     if not os.path.isfile(filename):
#         df.to_csv(filename, mode='a', index=False, header=True)
#     else:
#         df.to_csv(filename, mode='a', index=False, header=False)
#
# def metrics_boxplot(df: pd.DataFrame,
#                     x_var: str,
#                     y_var: str):
#     sns.boxplot(x=x_var, y=y_var, data=df)
#     plt.title(y_var + " by " + x_var)
#     plt.grid(which='major', axis='y', color='lightgray', linewidth=1)
#     plt.show()
#
#
# def metrics_boxplot_with_outlier_labels(df: pd.DataFrame, x_var: str, y_var: str, x_order=['TX', 'CA', 'VT']):
#     sns.set(
#         style="ticks",
#         rc={"figure.figsize": (12, 10),
#             "figure.dpi": 100})
#
#     fig, ax = plt.subplots()
#
#     # Set the order of the x-axis ticks
#     ax = sns.boxplot(x=x_var, y=y_var, data=df, ax=ax, order=x_order)
#
#     for i in range(df.shape[0]):
#         if (df[y_var].iloc[i] > df[y_var].quantile(0.95)) or (df[y_var].iloc[i] < df[y_var].quantile(0.05)):
#             ax.annotate(df.id[i], xy=(x_order.index(df[x_var].iloc[i]), df[y_var].iloc[i]), fontsize=12, color='red')
#
#     plt.title(y_var + " by " + x_var)
#     plt.grid(which='major', axis='y', color='lightgray', linewidth=1)
#
#     return fig
#
#
# def metrics_with_boxplot(df: pd.DataFrame, x_var: str, y_var: str, x_order=None):
#     sns.set(
#         style="ticks",
#         rc={"figure.figsize": (12, 10),
#             "figure.dpi": 100})
#
#     fig, ax = plt.subplots()
#
#     # Remove values that are outside the desired range
#     # Remove outliers based on quantiles
#     q_low = df[y_var].quantile(0.05)
#     q_high = df[y_var].quantile(0.95)
#     data = df[(df[y_var] >= q_low) & (df[y_var] <= q_high)]
#
#     # Set the order of the x-axis ticks
#     if x_order:
#         ax = sns.boxplot(x=x_var, y=y_var, data=data, ax=ax, order=x_order)
#     else:
#         ax = sns.boxplot(x=x_var, y=y_var, data=data, ax=ax)
#
#     plt.title(y_var + " by " + x_var)
#     plt.grid(which='major', axis='y', color='lightgray', linewidth=1)
#
#     # Add median values to the boxplot
#     if x_order:
#         medians = data.groupby([x_var])[y_var].median()[x_order].values
#         pos = range(len(medians))
#         for tick, label in zip(pos, ax.get_xticklabels()):
#             ax.annotate(f"Median: {medians[tick]:.2f}", xy=(tick, medians[tick]), xytext=(0, 15),
#                         textcoords='offset points', ha='left', va='top', fontsize=12, color='red', weight='bold')
#     else:
#         medians = data.groupby([x_var])[y_var].median().values
#         pos = range(len(medians))
#         for tick, label in zip(pos, ax.get_xticklabels()):
#             ax.annotate(f"Median: {medians[tick]:.2f}", xy=(tick, medians[tick]), xytext=(0, 15),
#                         textcoords='offset points', ha='left', va='top', fontsize=12, color='red', weight='bold')
#
#     return fig


if __name__ == '__main__':

    load_dotenv()

    # From env load the path of the csv files
    result_csv = os.getenv('RES_CSV')

    file_list = os.listdir(result_csv)

    # Create an empty dataframe
    error_df = pd.DataFrame()

    for file in file_list:
        # if file contain the string 'df' and end with '.csv' open it in a dataframe
        if 'df' in file and file.endswith('.csv'):
            df = pd.read_csv(result_csv + '\\' + file)

            col_name = ['timestep', 'Treal', 'Tpred']

            # rename the columns of the dataframe
            df.columns = col_name

            # Selected building after Kathryn's analysis
            selected_buildings = ['199613', '411001', '450491', '508889', '376570', '391597', '245723', '425540', '4421', '498771']
            id_map = {'199613': '1', '411001': '2', '450491': '3',
                      '508889': '4', '376570': '5',
                      '391597': '6', '245723': '7', '425540': '8',
                      '4421': '9', '498771': '10'}

            # if file contains one string of the selected_buildings plot
            if any(building in file for building in selected_buildings):
                # Plot the Treal vs Tpred of the first 14*24 timestep
                plt.figure(figsize=(12, 8))  # Adjusted figure size
                plt.plot(df['Treal'][:14 * 24], label='Energy+', linewidth=2)  # Increased line width
                plt.plot(df['Tpred'][:14 * 24], label='LSTM', linewidth=2)  # Increased line width
                plt.xlabel('Timestep', fontsize=14)  # Increased font size for x-axis label
                plt.ylabel('Temperature (°C)', fontsize=14)  # Increased font size for y-axis label
                # Get building ID from file name and map it to building name
                building_id = file.split('predictions')[-1].split('.')[0]
                building_name = id_map.get(building_id, 'Unknown')
                plt.title('Building ' + building_name, fontsize=16)  # Increased font size for title
                plt.grid(which='both', color='lightgray', linewidth=0.5)  # Added gridlines
                plt.locator_params(axis='y', nbins=5)  # Adjusted number of y-axis tick marks
                plt.xticks(fontsize=12)  # Increased font size for x-axis tick labels
                plt.yticks(fontsize=12)  # Increased font size for y-axis tick labels
                plt.legend(fontsize=12)  # Increased font size for legend
                plt.tight_layout()  # Improved layout spacing
                plt.legend()
                # Save each plot using the file name
                plt.savefig(result_csv + '\\plot\\' + file.split('.')[0] + '.png')

            # Extract the string after the last 'predictions' and before the first '.' from file
            building_id = file.split('predictions')[1].split('.')[0]

            # Evaluate the error between Treal and Tpred
            error_df[str(building_id)] = df['Treal'] - df['Tpred']

    # Create a ridge plot of the distribution of each column of the error_df
    # Melt the error dataframe to have each row as a separate observation
    melted_error_df = error_df.melt(var_name='Building', value_name='Error')

    plt.figure(figsize=(10, 7))

    # Create the joyplot with a beautiful colormap
    joyplot(melted_error_df, column='Error', by='Building', colormap=cm.hsv, alpha=0.7, linewidth=1)
    # Set plot labels and title
    plt.xlabel('Difference between Real and Predicted Temperature (°C)')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Errors for Each Building')
    plt.axvspan(-5, 5, color='lightgreen', alpha=0.2, zorder=0)
    # add a vertical line at 0
    plt.axvline(x=0, color='darkgreen', linestyle='--', linewidth=1)

    # Cut the plot at -5 and 5
    plt.xlim(-5, 5)

    # Save the plot
    plt.savefig(result_csv + '\\ridge_plot.png')



