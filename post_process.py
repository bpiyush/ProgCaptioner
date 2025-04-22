import os
import json
import glob


def process_response(response_dir):
    files = glob.glob(f"{response_dir}/*.json")
    os.makedirs(response_dir.replace('output', 'output_processed'), exist_ok=True)
    for file in files:
        output_file = file.replace('output', 'output_processed')
        with open(file, 'r') as f:
            data_list = json.load(f)
        for data_dict in data_list:
            response_list = data_dict['response0'].split('<Frame')[1:]
            assert len(response_list) == data_dict['n_frames']
            response_list = [r.replace('>', '')[3:].replace('*', '').strip() for r in response_list]
            data_dict['response0'] = response_list
        with open(output_file, 'w') as f:
            json.dump(data_list, f, indent=4)
        print(f"Processed {file} and saved to {output_file}")


def html_head():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Visualization with Predictions</title>
    <style>
        .label {
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .image-row {
            display: flex;
            flex-wrap: nowrap; /* No wrapping, all images in a single row */
            margin-bottom: 20px;
        }
        .image-container {
            text-align: center;
            margin: 10px;
            flex: 1 1 calc(25% - 20px); /* 4 images per row */
            box-sizing: border-box;
        }
        img {
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        .predictions {
            margin-top: 5px;
            text-align: left;
        }
    </style>
    </head>
    <body>
    """
    
    
def viz_data_with_pred(file_path):
    html_content = html_head()
    with open(file_path, 'r') as f:
        data_list = json.load(f)
    print(f"Reading {len(data_list)} from {file_path}")
    
    for data_dict in data_list:
        idx = data_dict['idx']
        image_files = data_dict['image_files']
        response_list = data_dict['response0']
        label = data_dict['action_label'] if 'action_label' in data_dict else None
        html_content += f'<div class="label">{idx}: {label}</div>\n'
        html_content += '<div class="image-row">\n'
        for i, image_file in enumerate(image_files):
            html_content += f"""
                <div class="image-container">
                <img src="../../{image_file}" alt="Image">
            """
            html_content += f"""<div class="predictions">{response_list[i]}</div>"""
            html_content += "</div>\n"
        html_content += "</div>\n"
    html_content += """</body></html>""" 
    
    save_path = f"data/viz_html/{os.path.basename(file_path).replace('.json', '')}.html"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as html_file:
        html_file.write(html_content)
    print(f"Saved visualization to {save_path}")
    
    
if __name__ == "__main__":
    # process_response('data/data_files/output')
    viz_data_with_pred('data/data_files/output_processed/coin_valhs_seq.json')