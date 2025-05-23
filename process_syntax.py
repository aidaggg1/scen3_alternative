import _io
import json
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import paho.mqtt.client as mqtt
import subprocess, time, requests, yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, Union, Type

#function for loading the YAML file
def load_syntax(path):
    with open(path, 'r') as file:
        syntax = yaml.safe_load(file)
    return syntax

#function for reading input(s) of the syntax file
def read_input(input_data):
    num_inputs = len(input_data.keys())
    info_input = {}  #saving the info about inputs

    #for each input get the information
    for i in range(num_inputs):
        input_name = list(input_data.keys())[i]  #getting the name of the inputs
        data = input_data[input_name]  #getting every data inside one input
        input_type = data["type"]

        input_info = {
            "name": input_name,
            "type": input_type
        }

        #if input is an image - threshold doesn't exist
        if "threshold" in data:
            input_threshold = data["threshold"]
            input_info["threshold"] = input_threshold
        #neither default
        if "default" in data:
            input_default = data["default"]
            input_info["default"] = input_default

        #saving in each iteration the data read
        info_input[f"input_{i + 1}"] = input_info

    return info_input

def get_introduce_inputs(syntax):
    all_inputs = {}
    all_orders = {}
    introduce_inputs = {}

    scenario = syntax["type"]
    num_models = len(syntax["models"].keys())

    for z in range(num_models):
        model_name = list(syntax.get("models", {}).keys())[z]

        if "input" in syntax["models"][model_name]:
            input = read_input(syntax["models"][model_name]["input"])  # getting all inside inputs
        else:
            input = read_input(syntax["common_input"])  # if there is a common structure
            
        all_inputs[model_name] = input  # store the input

        if scenario == "dependent":
            order = syntax["models"][model_name]["order"]
            all_orders[model_name] = order

    if scenario == "dependent":
        filter_orders = {key: value for key, value in all_orders.items() if str(value).startswith('1')}
        introduce_inputs = {key: all_inputs[key] for key in filter_orders.keys()}

    else:
        introduce_inputs = all_inputs

    return introduce_inputs

#function for reading servitization technology
def read_servitization_technology(syntax, model_name=None):
    if "common_servitization_technology" in syntax:
        servitization_technology_data = syntax["common_servitization_technology"]
    else:
        servitization_technology_data = syntax["models"][model_name]["servitization_technology"]

    serv_type = servitization_technology_data["type"]

    servitization_technology_name = list(serv_type.keys())[0]  #getting API or pub-sub ---- it's serv_type's key
    data_serv = serv_type[servitization_technology_name]  #getting everything inside API
    serv_name = data_serv["type"]  #getting RESTful or MQTT

    serv_info = {
        "name": servitization_technology_name, #API
        "type": serv_name  #Restful
    }

    #if it is an API --> url
    if servitization_technology_name == "API":
        endpoint_data = data_serv["endpoint"]
        url = endpoint_data["url"]
        port = endpoint_data["port"]
        endpoint_name = endpoint_data["name_endpoint"]
        endpoint_method = endpoint_data["method"]
        full_url = f"{url}{port}{endpoint_name}"
        serv_info["full_url"] = full_url  #save the whole url
        serv_info["endpoint_method"] = endpoint_method

    #if it is pub-sub --> broker + topic
    elif servitization_technology_name == "pub-sub":
        broker = data_serv["broker"]
        topic_in = data_serv["topic_input"]
        topic_out = data_serv["topic_output"]
        serv_info["broker"] = broker
        serv_info["topic_in"] = topic_in
        serv_info["topic_out"] = topic_out

    return serv_info

def open_image_file(image_path: str):
    #verify the path
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.getcwd(), image_path)

    #verify that the image exists
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"Image does not exist at {image_path}")

    return open(image_path, "rb")


def prepare_model_inputs(syntax, models_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    #save the response of each model in responses to give the final output
    responses = {}
    responses_sec = {} #only for dependent scenario

    for model_name, inputs in models_data.items():
        inputs_model = {}

        #iterate over the inputs of each model
        for input_name, value in inputs.items():
            #if that input is an image
            if isinstance(value, str) and value.endswith(('.jpg', '.jpeg', '.png', '.gif')):  #if this inputs is an image
                img_file = open_image_file(value)  #open the image from its path
                inputs_model[input_name] = img_file
            else:
                inputs_model[input_name] = value

        #I send to each model its own inputs, first get if this model is and api or pub-sub
        if syntax["type"] == "dependent":
            responses_sec.update(communicate_model(syntax, inputs_model, model_name))
        else:
            responses[model_name] = communicate_model(syntax, inputs_model, model_name)

    #if not dependent scenario, I have already been communicated with all the models - so I can call now prepare_output
    if syntax["type"] != "dependent":
        # call to prepare_output with all the responses from all models
        prepare_output(syntax, responses)

    #if the scenario is dependent, the output of one model is the input of the next one. So the response of the first has to be sent to the second
    else:
        all_orders = {}
        for model in syntax["models"]:
            order = syntax["models"][model]["order"]
            all_orders[model] = order

        #get how many models with order that isn't one are there
        not_one_orders = {key: value for key, value in all_orders.items() if not str(value).startswith('1')}

        response = responses_sec
        #for each model that is not order 1 --> send to the next model the previous response
        for not1 in not_one_orders:
            response = communicate_model(syntax, response, not1)

        #once I have all the responses, I write an output to the user
        prepare_output(syntax, response)

#this function manages the communication between C --> Model. Sends the input and receives the prediction
def communicate_model(syntax, inputs: dict, model_name: str):
    all_responses = {} #to store the response of each model (predictions)
    files = {}  # to store the files - images
    json_data = {}  # to store all the other type of inputs

    #split files from other input types
    for key, value in inputs.items():
        if isinstance(value, _io.BufferedReader):  # if this inputs is an already opened file (because in open_image_file the image is being opened)
            if value.name.endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"  #if the image has the extension .jpg
            elif value.name.endswith('.png'):
                mime_type = "image/png" #extension png
            files = {"file": (value.name, value, mime_type)}
        else:
            json_data[key] = value  #if it is not file, add it as json data

    #first I have to know how is the servitization_technology of this model
    serv = read_servitization_technology(syntax, model_name)
    type_model = serv["name"]

    #if the actual model is an api
    if type_model == "API":
        method = serv["endpoint_method"].lower()
        if method in ["get", "post", "put", "delete"]:
            try:
                #if there are no files, "files" is not sent. Same with "json_data"
                response = requests.request(method, serv["full_url"], files=files if files else None, json=json_data if json_data else None)
                #when a request is made, the model responses a prediction

                #in case the response of the isn't' a json (in LLM it isn't)
                try:
                    resp_content = response.json()
                except json.JSONDecodeError:
                    resp_content = {"response":response.text}

                print(f"Response in API (process_syntax.py): {response.status_code} - {response.json()}")

                #the llm gives me the answer as a str, so I convert into a dict
                if isinstance(resp_content, str):
                    # resp_content = json.loads(resp_content)
                    # all_responses.update(resp_content)

                    json_match = re.search(r'\{.*\}', resp_content, re.DOTALL)
                    if json_match:
                        try:
                            resp_content = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            print("Not possible to decode the json.")
                            resp_content = {"response": resp_content}
                        else:
                            resp_content = {"response": resp_content}

                all_responses.update(resp_content)

            except Exception as e:
                print(f"Error when sending the request: {e}")

        else:
            print(f"Error, HTTP method '{method}' not valid.")

    #if the model is pub-sub
    elif type_model == "pub-sub":       #the orchestrator publishes is the topic for input and the el model listenes on it
                                        #the model publishes in the topic for output and the orchestrator listens on it
        broker = serv["broker"]
        topic_in = serv["topic_in"]
        topic_out = serv["topic_out"]
        def on_message(client, userdata, msg):
            print(f"Received message from MQTT broker (process.py): {msg.payload.decode()}")
            response = msg.payload.decode()
            response_dict = json.loads(response)
            all_responses.update(response_dict)
        try:
            client = mqtt.Client()
            print(f"Connecting to MQTT broker (process.py): {broker}")
            client.connect(broker, 1883, 60)
            client.on_message = on_message
            client.subscribe(topic_out) #listens on the topic which the model publishes
            client.loop_start()

            #publish the message
            if json_data:
                payload = json.dumps(json_data)
                client.publish(topic_in, payload=payload)
                print(f"Message sent to MQTT broker (process.py) '{broker}' in topic '{topic_in}': {json_data}")

            else:
                print("No message to send to MQTT.")

            timeout = 5
            start_time = time.time()
            while not all_responses and (time.time() - start_time) < timeout:
                time.sleep(0.5)
            client.loop_stop()

            if not all_responses:
                print("Error: responses not received from mqtt (process.py")

        except Exception as e:
            print("Connection to broker failed" + {e})

    else:
        return (f"Error, wrong protocol {type_model}. Choose API or pub-sub")

    return all_responses


def validate_output_threshold(syntax, model_name, value):
    #if there is common output
    if "common_output" in syntax:
        output_data = syntax["common_output"]
    else:
        output_data = syntax["models"][model_name]["output"]

    #if there is a threshold for the specified model, validate it
    if "threshold" in output_data:
        output_threshold = output_data["threshold"]
        lower_bound, upper_bound = map(int, output_threshold.split('-'))
        if not (lower_bound <= value <= upper_bound):
            print("The value calculated by the model is out of the specified threshold")
        else:
            print("The value calculated by the model is inside the specified threshold")
    else:
        return  #if there is not specified an output threshold do nothing


#this function gives the final output which the user will read to understand what is happening in the system
def prepare_output(syntax, responses: Union[Dict[str, Any], str]):
    scenario = syntax["type"]
    print(f"Since you are in a {scenario} scenario type, this is the information about the system")

    if scenario == "dependent":
        print(f"The final response is {responses}")

    else:
        for model_name, response in responses.items():
            #in case the model returns its answer in different ways

            #if it is a list
            if isinstance(response, list) and len(response) > 0:
                first_element = response[0]

                if isinstance(first_element, dict):
                    prediction_data = first_element
                else:
                    print(f"Data structure in the model {model_name} is not valid")
                    continue  # skip this model

                for key, value in prediction_data.items():
                    print(f"The model '{model_name}' response is '{key}: {value}'")
                    validate_output_threshold(syntax, model_name, value)

            #if it is a dict
            elif isinstance(response, dict):
                for key, value in response.items():
                    print(f"The model '{model_name}' response is '{key}: {value}'")
                    validate_output_threshold(syntax, model_name, value)

            #if it is a str
            elif isinstance(response, str):
                print(f"The model '{model_name}' response is '{response}'")
                validate_output_threshold(syntax, model_name, response)

            else:
                print(f"Error. The model '{model_name}' response is not valid.")

        if scenario == "alternative":
            av = calculate_weighted_average(syntax, responses)
            print(f"But as it as an alternative scenario, the average calculated is: {av}")

#for scenario == alternative
def calculate_weighted_average(syntax, responses):
    sum_weights = 0.0
    stored_mult = 0.0

    for model_name, resp in responses.items():
        for key, value in resp.items():
            weight = get_weight(syntax, model_name)
            stored_mult += weight * value
            sum_weights += weight

    if sum_weights > 0:
        return round(stored_mult / sum_weights, 4)
    return None

def get_weight(syntax, model_name):
    return syntax["models"][model_name]["weight"]

def execute_models(scenario, models_data):
    all_orders = {}
    model_paths = []

    #get the path of the models to execute them
    for m in models_data:
        full_model_path = models_data[m]["path"]
        filename = os.path.basename(full_model_path)  # to get the .py
        model_paths.append(filename)

        if scenario == "dependent":
            #get prority
            order = models_data[m]["order"]
            all_orders[filename] = order

    #execute the files of the models
    if scenario in ["holistic", "alternative"]:
        # execute every model
        for m in model_paths:
            subprocess.Popen(["python3", m])
            time.sleep(3)

    elif scenario == "dependent":
        desc_order = sorted(all_orders.items(), key=lambda x: x[1], reverse=True)

        # if I have the servicesA, B, C. With orders 1,2,3. first execute C, then B, finally A -- so order in descended
        for m, order in desc_order:
            subprocess.Popen(["python3", m])
            time.sleep(3)  # wait 3 secs


#to show in the api of the orchestrator the name of the inputs and not all their content (type, threshold no)
def get_inputs_names(introduce_inputs: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    all_inputs = {}
    for model in introduce_inputs.values():
        for input_data in model.values():
            all_inputs[input_data["name"]] = input_data
    return all_inputs

def generate_dynamic_model(inputs: Dict[str, Dict[str, Any]]) -> Type[BaseModel]:
    fields = {}
    defaults = {}

    for key, value in inputs.items():
        name = value["name"]
        field_type = value["type"]

        if field_type == "float":
            fields[name] = float
        elif field_type == "int":
            fields[name] = int
        else:
            fields[name] = str

        if "default" in value:
            defaults[name] = value["default"]

    model_attrs = {"__annotations__": fields, **defaults}
    return type("DynamicBaseModel", (BaseModel,), model_attrs)

#to validate the inputs introduced, check if they are inside the specified threshold
def validate_input_thresholds(data: Dict[str, Any], all_inputs: Dict[str, Dict[str, Any]]):
    errors = {}

    for key, value in data.items():
        if key in all_inputs and "threshold" in all_inputs[key]:
            min_val, max_val = map(float, all_inputs[key]["threshold"].split("-"))

            if not (min_val <= value <= max_val):
                errors[key] = f"The value entered {value} is out of the threshold specified: ({min_val}-{max_val})"

    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})


def main():
    syntax = load_syntax("syntax3.yml")  #here the name of the syntax file has to be changed

    scen_type = syntax["type"]

    if scen_type not in ["holistic", "dependent", "alternative"]:
        print("Error: the scenario type you entered is not an option. Select holistic, dependent or alternative")
        return

    execute_models(scen_type, syntax["models"])

    introduce_inputs = get_introduce_inputs(syntax)
    all_inputs = get_inputs_names(introduce_inputs)

    DynamicBaseModel = generate_dynamic_model(all_inputs)

    app = FastAPI()

    @app.post("/validate_data/")
    def validate_data(data: DynamicBaseModel):
        try:
            introduced_data = data.model_dump()
            validate_input_thresholds(introduced_data, all_inputs)

            first_models_data = {}
            for model_name, inputs in introduce_inputs.items():
                first_models_data[model_name] = {key: value for key, value in introduced_data.items() if
                                                 key in [input_data['name'] for input_data in inputs.values()]}

            prepare_model_inputs(syntax, first_models_data)

            return {"message": "Data is correct", "data": first_models_data}

        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

    if __name__ == '__main__':
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=9009)

#call main
main()
