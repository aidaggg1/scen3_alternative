import _io
import json
#from statistics import median_grouped
import base64
import os

import paho.mqtt.client as mqtt
import subprocess, time, requests, yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, create_model, ValidationError
from typing import Dict, Any, Union, Optional, Type

#function for loading the yaml file
def load_syntax(path):
    with open(path, 'r') as file:
        syntax = yaml.safe_load(file)
    return syntax

#function for reading input(s)
def read_input(input_data):
    num_inputs = len(input_data.keys())
    info_input = {}  # here saving the info about inputs

    #using a FOR for each input
    for i in range(num_inputs):
        input_name = list(input_data.keys())[i]  #getting Temperature...Vibration...
        data = input_data[input_name]  #getting everything inside Temperature {'type': 'float', 'threshold': '50.0-150.0', 'default': 80.0})
        input_type = data["type"]

        input_info = {
            "name": input_name,
            "type": input_type
        }

        #if input is an image - threshold doesnt exist
        if "threshold" in data:
            input_threshold = data["threshold"]
            input_info["threshold"] = input_threshold
            # neither default
        if "default" in data:
            input_default = data["default"]
            input_info["default"] = input_default

        #saving in each iteration the data read
        info_input[f"input_{i + 1}"] = input_info

    return info_input

def get_scenario(syntax):
    scenario = syntax["type"]
    if scenario == "holistic":
        return 1
    elif scenario == "dependent":
        return 2
    elif scenario == "alternative":
        return 3

def get_introduce_inputs(syntax):
    all_inputs = {}
    all_priorities = {}
    introduce_inputs = {}

    scenario = get_scenario(syntax)
    num_models = len(syntax["models"].keys())

    for z in range(num_models):
        model_name = list(syntax.get("models", {}).keys())[z]

        # READ INPUT(s)
        if "input" in syntax["models"][model_name]:
            input = read_input(syntax["models"][model_name]["input"])  # getting all inside inputs
        else:
            input = read_input(syntax["common_input"])  # if there is a common structure
        all_inputs[model_name] = input  # store the input

        if scenario == 2:
            priority = syntax["models"][model_name]["priority"]
            all_priorities[model_name] = priority

    if scenario == 1:
        introduce_inputs = all_inputs

    elif scenario == 2:
        filter_priorities = {key: value for key, value in all_priorities.items() if str(value).startswith('1')}
        introduce_inputs = {key: all_inputs[key] for key in filter_priorities.keys()}

    elif scenario == 3:
        introduce_inputs = all_inputs

    # if there is common_input
    print(f"introduce_inputs es ::: {introduce_inputs}")
    return introduce_inputs


#function for reading output
def read_output(output_data):
    output_name = output_data["name"]
    output_type = output_data["type"]

    output_info = {
        "name": output_name,
        "type": output_type
    }

    #if threshold isnt especified dont save this field
    if "threshold" in output_data:
        output_threshold = output_data["threshold"]
        output_info["threshold"] = output_threshold

    return output_info
    #print(" Output: ", output_info) #return en vez de print??

#function for reading environment
#he cambiado el parametro por el nombre del modelo
#DESPUES CONTEMPLAR EL CASO EN EL QUE HAYA COMMON ENVIRONMENT   !!!!!!!!!!!!
def read_environment(model_name=None):
    if "common_environment" in syntax:
        environment_data = syntax["common_environment"]
    else:
        environment_data = model_name["environment"]
    env_type = environment_data["type"]  #inside type  {'API', 'type': 'RESTful', 'full_url': 'http://localhost:5000/receive', 'endpoint_method': 'POST'}

    environment_name = list(env_type.keys())[0]  #getting API or pub-sub ---- it's env_type's key
    data_env = env_type[environment_name]  # getting everything inside API
    env_name = data_env["type"]  #getting RESTful or MQTT

    env_info = {
        "name": environment_name, #API
        "type": env_name  #Restful
    }

    #if it is an API --> endpoint
    if environment_name == "API":
        endpoint_data = data_env["endpoint"]
        url = endpoint_data["url"] #getting "http://localhost:"
        port = endpoint_data["port"]
        endpoint_name = endpoint_data["name_endpoint"]  #getting predict
        endpoint_method = endpoint_data["method"]  #getting POST
        full_url = f"{url}{port}{endpoint_name}"
        env_info["full_url"] = full_url
        env_info["endpoint_method"] = endpoint_method

    #if it is pub-sub --> broker + queue
    elif environment_name == "pub-sub":
        broker = data_env["broker"]
        queue = data_env["queue"]
        env_info["broker"] = broker
        env_info["queue"] = queue

    return env_info

#read the image path, open it and send to the model
def open_image_file(image_name: str):
    current_directory = os.getcwd()  #get the current directory
    image_path = os.path.join(current_directory, f"{image_name}")

    # Verificar si la imagen existe
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"Image {image_name} does not exist at {image_path}")

    # problemas: si uso with open, el archivo se abre para leerlo pero cuando el bloque termina, se cierra. entonces luego en requests estoy intentando procesar una imagen en la api de clasificación que se ha cerrado
    # en vez de usar with open voy a hacer solo open
    return open(image_path, "rb")

def prepare_model_inputs(models_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    #save the response of its model in responses to give the final output
    responses = {}
    for model_name, inputs in models_data.items():
        inputs_model = {}

        #iterate over the inputs of its model
        for input_name, value in inputs.items():
            #si este input es una imagen
            if isinstance(value, str) and value.endswith(('.jpg', '.jpeg', '.png', '.gif')):  #if this inputs is and image
                img_file = open_image_file(value)  #open the image from its path
                inputs_model[input_name] = img_file
            else:
                inputs_model[input_name] = value

        #I send to its models its own inputs , first get if this model is and api or pub-sub
        #syntax es una variable definida fuera de cualquier función, puedo acceder a ella
        env = read_environment(syntax["models"][model_name])  # el environment del modelo de esta iteracion - tengo {'name': 'API', 'type': 'RESTful', 'full_url': 'http://localhost:5000/predict', 'endpoint_method': 'POST'}
        if env["name"] == "API":
            url_model = env["full_url"]
            method_model = env["endpoint_method"]
            responses[model_name] = send_to_model(inputs_model,  env["name"], url_api = url_model, method_api = method_model)
        elif env["name"] == "pub-sub":
            broker_model = env["broker"]
            queue_model = env["queue"]
            responses = send_to_model(inputs_model, env["name"], broker_mqtt = broker_model, queue_mqtt = queue_model)

    #call to prepare_output with all the responses from all models
    prepare_output(responses)

'''
type_model is api or pub-sub
url_api will only especified if type_model is an api
broker_mqtt and queue_mqtt will only especified if type_model is pub-sub
'''
def send_to_model(inputs : dict, type_model: str, url_api: str=None, method_api: str = None, broker_mqtt: str =None, queue_mqtt: str =None):
    print(f"en send to model {inputs} {type_model} {url_api} {method_api}")

    all_responses = [] #to store al the responses (predictions)

    #if the actual model is an api
    if type_model == "API":
        method = method_api.lower()
        if method in ["get", "post", "put", "delete"]:
            files = {}  #to store the files - images
            json_data = {}  #to store all the other type of inputs

            #split files from other input types
            for key, value in inputs.items():
                if isinstance(value, _io.BufferedReader):  #if this inputs is an already opened file (because in open_image_file the image is being opened)
                    if value.name.endswith(('.jpg', '.jpeg')):
                        mime_type = "image/jpeg"  # MIME type para JPEG
                    elif value.name.endswith('.png'):
                        mime_type = "image/png"
                    files = {"file": (value.name, value, mime_type)}
                else:
                    json_data[key] = value  #if not file, add it as json data

            try:
                #if there are no files, it is not send. Same with json data
                response = requests.request(method, url_api, files=files if files else None, json=json_data if json_data else None)
                all_responses.append(response.json())
                print(f"Response in API (process_syntax.py): {response.status_code} - {response.json()}")
            except Exception as e:
                print(f"Error when sending the request: {e}")

        else:
            print(f"Error, HTTP method '{method}' not valid.")

        return all_responses


    #   HACER !!!!!!
    elif type_model == "pub-sub":
        return

    else:
        return (f"Error, wrong protocol {type_model}. Choose API or pub-sub")

#this function gives the final output which the user will read to understand what is happing in the system
def prepare_output(responses: Dict[str, Any]):
    scenario = syntax["type"]
    print(f"Since you are in a {scenario} scenario type, this is the information about the system")
    for model_name, response in responses.items():
        if isinstance(response, list) and len(response) > 0:
            prediction_data = response[0]  # Accede al primer (y único) diccionario dentro de la lista

            for key, value in prediction_data.items():
                print(f"The model '{model_name}' response is '{key}: {value}'")

        #if it is an alternative, give the weighted average
        #TODO
        if scenario == "alternative":

            print(f"But as it as an alternative scenario, the average calculated is: ")

        else:
            print(f"Error: The model '{model_name}' response is not valid.")

#if we are in an alternative scenario - have to calculate the weighted average
def calculate_weighted_average(predictions, weights):
    """Calcula la media ponderada de las predicciones usando los pesos definidos"""
    total_weight = sum(weights.values())
    weighted_sum = sum(predictions[model] * weights[model] for model in weights)

    return weighted_sum / total_weight if total_weight > 0 else None

#process all the file
def process(syntax):
    project_name = syntax.get("project_name", "Unknown project") #getting project_name
    scenario_type = syntax["type"]

    valid_scenarios = ["holistic", "dependent", "alternative"]
    if scenario_type not in valid_scenarios:
        print("Error: the scenario type you entered is not an option. Select holistic, dependent or alternative")
        return

    print("Project name:", project_name)
    print("Scenario type:", scenario_type)

    #using a FOR for each model
    num_models = len(syntax["models"].keys())
    all_envs ={}
    env_data = {}
    all_priorities = {}
    all_weights = {}
    all_model_names = []

    for z in range (num_models):
        #READ MODEL NAME                        #ir guardando todos los model_name
        model_name = list(syntax.get("models", {}).keys())[z]  #getting model_name (machine_failure)
        print("Model name:", model_name)

        #READ OUTPUT
        if "output" in syntax["models"][model_name]:
            read_output(syntax["models"][model_name]["output"]) #getting all inside inputs
        else:
            read_output(syntax["common_output"]) #if there is a common structure


        #READ ENVIRONMENT
        env_desc = read_environment(syntax["models"][model_name])

        #obtengo si cada modelo está en una api, pub/sub...
        env_type = env_desc["name"]  #me quedo solo con el name (api o pub sub)
        all_envs[model_name] = env_type  #esto guarda en la última iteración {'serviceA-brokerMAL.py': 'pub-sub', 'serviceB.py': 'API', 'full_url': 'http://localhost:5000/receive'}
        all_model_names.append(model_name)

        env_data[model_name] = env_type
        if env_type == "API":
            env_data["full_url"] = env_desc["full_url"]

        if scenario_type == "dependent":
            #get prority
            priority = syntax["models"][model_name]["priority"]
            all_priorities[model_name] = priority       #en all_priorities tengo en la última iteración {'serviceA.py': 1, 'serviceB.py': 2}

        elif scenario_type == "alternative":
            weight = syntax["models"][model_name]["weight"]
            all_weights[model_name] = weight

    print(f" weights: {all_weights}")

            # !!!!! EN VEZ DE HACER PRINTS, PODRIA GUARDARLO EN UNA VARIABLE, O HACER RETURN EN FORMA DE DICCIONARIO, RETURN DE UNA VARIABLE


    if scenario_type == "holistic":
        # ejecutar cada modelo
        for m in all_model_names:
            print(f"Iniciando el servicio {m}")
            subprocess.Popen(["python3", f"{m}.py"])
            time.sleep(3)

    if scenario_type == "dependent":
        #inputs will be only the models with prorities 1, 1.1, 1.2... (starting with 1)
        filter_priorities = {key: value for key, value in all_priorities.items() if str(value).startswith('1')}
        #introduce_inputs = {key: all_inputs[key] for key in filter_priorities.keys()}
        print("modelos de prioridad uno", filter_priorities)
        print("all envs", all_envs)

        #order from least pririty to most
        desc_priority = sorted(all_priorities.items(), key=lambda x: x[1], reverse=True)
        print("prioridades desc", desc_priority)

        # initialize the services before starting the communication
        # if I have the servicesA, B, C. With priorities 1,2,3. first execute C, then B, finally A
        # thats why all_priorities is ordered descendent
        for m, priority in desc_priority:
            print(f"Iniciando el servicio {m}")
            subprocess.Popen(["python3", f"{m}.py"])
            time.sleep(3)  # wait 3 secs

        #if there is more than one model with priority 1 :
        for model1 in filter_priorities.keys():
            first_protocol = all_envs[model1]

            #1. option -- if the protocol the user has to interact with is a pub-sub :
            if first_protocol == "pub-sub":
                model1_broker = syntax["models"][model1]["environment"]["type"]["pub-sub"]["broker"]
                model1_queue = syntax["models"][model1]["environment"]["type"]["pub-sub"]["queue"]
                print("To execute " + model1 + "with priority 1, which uses " + first_protocol + " communicate with broker '" + model1_broker + "' and publish in queue '" + model1_queue + "'")

                #get the next service in desc_priorities
                next_service = desc_priority[0][0]  #the next service with less priority
                next_protocol = all_envs.get(next_service)

                if next_protocol == "API":
                    full_url_B = env_data.get("full_url") if env_data.get(next_service) else None
                    next_method = syntax["models"][next_service]["environment"]["type"]["API"]["endpoint"]["method"]

                    #print(f"valores a communicate {first_protocol} {next_protocol} {model1_broker} {model1_queue} {full_url_B} {next_method}")

                    communicate_dep(
                        protocol_A=first_protocol,
                        protocol_B=next_protocol,
                        broker_A=model1_broker,
                        queue_A=model1_queue,
                        url_B=full_url_B,
                        method_B=next_method
                    )

                elif next_protocol == "pub-sub":
                    next_broker = syntax["models"][next_service]["environment"]["type"]["pub-sub"]["broker"]
                    next_queue = syntax["models"][next_service]["environment"]["type"]["pub-sub"]["queue"]

                    communicate_dep(
                        protocol_A=first_protocol,
                        protocol_B=next_protocol,
                        broker_A=model1_broker,
                        queue_A=model1_queue,
                        broker_B=next_broker,
                        queue_B=next_queue
                    )

            #2. option -- if the protocol the user has to interact with is an API :
            if first_protocol == "API":
                # get the next service in desc_priorities
                next_service = desc_priority[0][0]  # here "serviceB"
                next_protocol = all_envs.get(next_service)

                full_url_A = env_data.get("full_url") if env_data.get(next_service) else None
                model1_method = syntax["models"][model1]["environment"]["type"]["API"]["endpoint"]["method"]
                print("To execute " + model1 + "with priority 1, which uses " + first_protocol + " executes in  '" + full_url_A + "' and method '" + model1_method + "'")

                if next_protocol == "API":
                    full_url_B = env_data.get("full_url") if env_data.get(next_service) else None
                    next_method = syntax["models"][next_service]["environment"]["type"]["API"]["endpoint"]["method"]

                    communicate_dep(
                        protocol_A=first_protocol,
                        protocol_B=next_protocol,
                        url_A=full_url_A,
                        method_A=model1_method,
                        url_B=full_url_B,
                        method_B=next_method
                    )

                elif next_protocol == "pub-sub":
                    next_broker = syntax["models"][next_service]["environment"]["type"]["pub-sub"]["broker"]
                    next_queue = syntax["models"][next_service]["environment"]["type"]["pub-sub"]["queue"]

                    #print(f"en communicate va {first_protocol}{next_protocol}{full_url_A}{model1_method}{next_broker}{next_queue}")

                    communicate_dep(
                        protocol_A=first_protocol,
                        protocol_B=next_protocol,
                        url_A=full_url_A,
                        method_A=model1_method,
                        broker_B=next_broker,
                        queue_B=next_queue
                    )

    elif scenario_type == "alternative":
        # ejecutar cada modelo
        for m in all_model_names:
            print(f"Iniciando el servicio {m}")
            subprocess.Popen(["python3", f"{m}.py"])
            time.sleep(3)

        #get weights
        for model, weight in all_weights.items():
            print(f"The model {model}, has the weight {weight}")


    #??????
    #devolver lo que he leido del yaml  ?!?!?!?!
    #info_sintaxis = f"Proyecto: {project_name} " + f"Modelo: {model_name} " + f"Entradas: {input_data}" #+ f"tipo entrada: {input1_type}" + f"umbral entrada: {input1_umbral}"
    #return info_sintaxis


syntax = load_syntax("syntax3.yml")   #change the syntax file name
process(syntax)


def get_inputs_names(introduce_inputs: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    all_inputs = {}
    for model in introduce_inputs.values():
        for input_data in model.values():
            all_inputs[input_data["name"]] = input_data
    return all_inputs

def generate_dynamic_model(inputs: Dict[str, Dict[str, Any]]) -> Type[BaseModel]:
    fields = {}
    defaults = {}
    print(f"inputs {inputs}")

    for key, value in inputs.items():
        name = value["name"]
        field_type = value["type"]

        if field_type == "float":
            fields[name] = float
        elif field_type == "int":
            fields[name] = int
        elif field_type == "string":
            fields[name] = str
        else:
            fields[name] = str

        if "default" in value:
            defaults[name] = value["default"]

    model_attrs = {"__annotations__": fields, **defaults}
    return type("DynamicBaseModel", (BaseModel,), model_attrs)

#to validate the inputs introduced, check if they are inside the specified threshold
def validate_thresholds(data: Dict[str, Any], all_inputs: Dict[str, Dict[str, Any]]):
    errors = {}

    for key, value in data.items():
        if key in all_inputs and "threshold" in all_inputs[key]: #si no hay threshold especificado no entra aquí -- NO?????
            min_val, max_val = map(float, all_inputs[key]["threshold"].split("-"))

            if not (min_val <= value <= max_val):
                errors[key] = f"The value entered {value} is out of the threshold specified: ({min_val}-{max_val})"

    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})


introduce_inputs = get_introduce_inputs(syntax)
all_inputs = get_inputs_names(introduce_inputs)

# Crear el modelo dinámico
DynamicBaseModel = generate_dynamic_model(all_inputs)

#so the user can introduce the inputs
app = FastAPI()

@app.post("/validate_data/")
def validate_data(data: DynamicBaseModel):
    #first validate the threshold
    try:
        data_dict = data.model_dump()
        validate_thresholds(data_dict, all_inputs)

        #dividir los inputs correspondientes a cada modelo
        models_data = {}
        for model_name, inputs in introduce_inputs.items():
            models_data[model_name] = {key: value for key, value in data_dict.items() if
                                      key in [input_data['name'] for input_data in inputs.values()]}
        prepare_model_inputs(models_data)

        return {"message": "Data is correct", "data": models_data}

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

#to run the api
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)

















#-----------------------------------------------------------------------------------------------------
#SEGURAMENTE TENGA QUE BORRAR TO.DO ESTO

#only in DEPENDENT scenario
#to manage the communication between 2 protocols
def communicate_dep(protocol_A, protocol_B, url_A = None, method_A = None, url_B = None, method_B=None, broker_A=None, queue_A=None, broker_B=None, queue_B=None):
    print(f"Communication from {protocol_A} to {protocol_B}")

    #if the first is pub-sub
    if protocol_A == "pub-sub":
        print(f"A uses {protocol_A}. You have to send a message to broker '{broker_A}' in the queue '{queue_A}'")

        def on_connect(client, userdata, flags, rc):
            print(f"Conectado con el código de resultado {rc}")
            client.subscribe(queue_A)

        def on_message(client, userdata, msg):
            received_message = msg.payload.decode()
            print(f"\nMenssage received in MQTT (process.py): {received_message}")

            if protocol_B == "API":
                print(f"B uses {protocol_B}. Sending a message to {url_B}")
                #to get the method
                method = method_B.lower()
                if method in ["get", "post", "put", "delete"]:
                    response = getattr(requests, method)(url_B, json={"message": received_message})
                else:
                    print(f"Error in HTTP method '{method}' ")
                    return

                print(f"Response in API B (process.py): {response.status_code} - {response.json()}")

            elif protocol_B == "pub-sub":
                client_publish = mqtt.Client()
                client_publish.connect(broker_B, 1883, 60)
                client_publish.publish(queue_B, payload=received_message)
                print(f"Message sent to MQTT broker (process.py) '{broker_B}' in queue '{queue_B}'")


        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(broker_A, 1883, 60)

        #client.subscribe(queue_A)
        print("\nConnecting to broker (process.py) {broker_A}")
        client.loop_forever()



    #if first is api
    elif protocol_A == "API":
        print(f"valores communicate: {protocol_A }{protocol_B} {url_A} {method_A} {broker_B} {queue_B}")
        print(f"A uses {protocol_A}. Waiting for the user to send a message to (process.py) '{url_A}'")

        # to get the method
        method = method_A.lower()
        if method in ["get", "post", "put", "delete"]:
                    #AQUI IRÁ EL MENSAJE QUE SE LEA DESDE LA API DEL CEREBRO ( en este json={} )
            response = getattr(requests, method)(url_A, json={}) #el json está vacío, lo tiene que coger de cuando se escribe en la url
            print(f"Response in API (process.py) : {response.status_code} - {response.json()}")
        else:
            print(f"Error in HTTP method '{method}' ")

        sending_message = response.json().get("message", None)
        if sending_message:
            print(f"Message received from API (process.py) : {sending_message}")
        else:
            print("No message field found in the API response.")

        if protocol_B == "pub-sub":
            try:
                client = mqtt.Client()
                print(f"Connecting to MQTT broker (process.py): {broker_B}")
                client.connect(broker_B, 1883, 60)

                # publish the message
                if sending_message:
                    client.publish(queue_B, payload=sending_message)
                    print(
                        f"Message sent to MQTT broker (process.py) '{broker_B}' in queue '{queue_B}': {sending_message}")
                else:
                    print("No message to send to MQTT.")

                client.loop_start()
                client.loop_stop()

            except Exception as e:
                print("Connection to broker failed" + {e})

    else:
        print("Unrecognised protocol. Cannot send the message.")