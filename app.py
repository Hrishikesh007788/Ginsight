from flask import Flask, render_template, request, redirect, send_file, url_for, json, send_from_directory
import os
import pandas as pd
from pandasai import SmartDatalake
from pandasai.llm import AzureOpenAI
from pandasai.llm.openai import OpenAI
from pandasai.helpers.openai_info import get_openai_callback
# from datetime import datetime
# from pandasai import Agent
from pandasai.middlewares import Middleware
from pandasai.responses.response_parser import ResponseParser
from tabulate import tabulate



app = Flask(__name__)

app.secret_key = 'your_secret_key_here'
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Get a list of all CSV and XLSX files in the "data" folder
data_files = [filename for filename in os.listdir(UPLOAD_FOLDER) if filename.endswith(('.csv', '.xlsx'))]

# Initialize an empty list to store DataFrames
data_frames = []
response = None
response2 = None
costing_data = []

class AggMiddleware(Middleware):
    def run(self, code: str) -> str:
        return f"matplotlib.use('agg')\n{code}"

for filename in data_files:
    # Construct the full path to the file
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Read each file into a DataFrame and append it to the data_frames list
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path, encoding="latin1")
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    data_frames.append(df)
   


# ..............OPENAI.................

# llm = OpenAI(api_token='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')


# ..............Azure OpenAI.............

llm = AzureOpenAI(
    api_token="ceeeca007c4545c3bef99e9bf4c11cfc",
    azure_endpoint='https://openai-so-gpt.openai.azure.com/',
    api_version="2023-05-15",
    deployment_name="gpt-35turbo-deployment"
)


def get_datasets():
    datasets_path = os.path.join('database', 'datasets.json')
    with open(datasets_path, 'r') as json_file:
        datasets_data = json.load(json_file)
        datasets = datasets_data.get("datasets", [])
    return datasets


def update_datasets_json(filename):
    datasets_path = os.path.join('database', 'datasets.json')
    with open(datasets_path, 'r') as json_file:
        datasets_data = json.load(json_file)
        datasets = datasets_data.get("datasets", [])
        datasets.append({"name": filename, "faqs": []})

    # Write the updated datasets list back to the JSON file
    with open(datasets_path, 'w') as json_file:
        json.dump({"datasets": datasets}, json_file)


class PandasDataFrame(ResponseParser):

    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        # Returns Pandas Dataframe instead of SmartDataFrame
        print(result)
        return result["value"]

@app.route('/download_excel', methods=['GET'])
def download_excel():

    excel_path = 'Output.xlsx'
    response2.to_excel(excel_path, index=False)

    # Send the file as a response to the client
    return send_file(excel_path, as_attachment=True)



@app.route('/', methods=['GET', 'POST'])
def index():
    global response, costing_data
    image_folder = os.path.join(app.static_folder, 'images')
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]
   
    if request.method == 'POST':
        try:
            
            df = SmartDatalake(data_frames, config={"llm": llm,"enforce_privacy" : True ,"save_charts" : True,"save_charts_path" : 'static\images',"custom_whitelisted_dependencies": ["tabulate"], "middlewares": [AggMiddleware()]})
          
            
            query = request.form['question']
            response2 = None
            query_tab = None
            query_lower = query.lower()
            # Load the Excel file containing keyword replacements
            keyword_replacements_file = 'data_dict.xlsx'
            df_keyword_replacements = pd.read_excel(keyword_replacements_file)

            # Convert the DataFrame to a dictionary
            keyword_replacements = df_keyword_replacements.set_index('keyword').to_dict()['replacement']
            for keyword, replacement in keyword_replacements.items():
                query_lower = query_lower.replace(keyword, replacement)
                
            
            keywords_to_check = ['visual', 'histogram', 'plot','visualize','visualise','chart','graph']
            for keyword in keywords_to_check:
                if keyword in query_lower:
                    query_lower = query_lower + ". Import seaborn. Display values as well"
                    with get_openai_callback() as cb:
                        response2 = df.chat(query_lower)

                        cb=cb
                    break
            else:
                query_tab = query_lower + ". Import tabulate. disable_numparse=True. dropna(axis=1)"
                with get_openai_callback() as cb:
                    response2 = df.chat(query_tab)
                    cb = cb
                    
            image_folder = os.path.join(app.static_folder, 'images')
            image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]
            
            return render_template('index.html', content=response2, data_files=data_files, datasets=get_datasets(), cb=cb, image_files=image_files)
        except Exception as e:
            print("ERRRRRRRRoOOORRR!!")
            print(e)
            return redirect(url_for('index'))
        
    return render_template('index.html', data_files=data_files, datasets=get_datasets(), image_files=image_files)

@app.route('/download/<filename>')
def download_image(filename):
    # Serve the image file for download
    image_folder = os.path.join(app.static_folder, 'images')
    return send_from_directory(image_folder, filename, as_attachment=True)



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # After saving the file, also append it to the data_frames list
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, encoding="latin1")
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filename, engine='openpyxl')
        data_frames.append(df)
        
        # Update the data_files list with the new filename
        data_files.append(file.filename)
        
        # Update the datasets JSON file with the new dataset name
        update_datasets_json(file.filename)

    return redirect(url_for('index'))

@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    image_folder = os.path.join(app.static_folder, 'images')
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]
    # Check if the image exists in the images folder
    image_path = os.path.join(app.static_folder, 'images', filename)
    if os.path.exists(image_path):
        # Delete the image file
        os.remove(image_path)
        # You may want to remove the image from the image_files list as well
        if filename in image_files:
            image_files.remove(filename)
    return redirect(url_for('index'))


if __name__ == '__main__':
      app.run(host = "0.0.0.0", port = 8000,debug=True)
