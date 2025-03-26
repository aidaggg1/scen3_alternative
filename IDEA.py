import yaml
import requests

'''
    1️⃣ Leer el YAML y extraer la configuración.
    2️⃣ Pedir los datos al usuario, validando tipo y umbrales.
    3️⃣ Enviar los datos a la API del cliente.
    4️⃣ Obtener las predicciones de los modelos.
    5️⃣ Aplicar la media ponderada con los pesos del YAML.
    6️⃣ Devolver el resultado final.
'''

def load_syntax(yaml_path):
    """Carga el YAML y devuelve su contenido como diccionario"""
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def get_user_input(common_input):
    """Solicita los inputs al usuario y los VALIDA según el YAML"""
    user_input = {}

    for key, details in common_input.items():
        while True:
            try:
                # Obtener el tipo de dato esperado
                expected_type = details["type"]
                default_value = details.get("default")

                # Pedir valor al usuario con opción de usar el valor por defecto POR CONSOLA
                user_value = input(f"Ingrese {key} ({expected_type}, default={default_value}): ") or default_value

                # Convertir al tipo correcto
                if expected_type == "integer":
                    user_value = int(user_value)
                elif expected_type == "number":
                    user_value = float(user_value)

                # Validar umbrales si se ha especificado
                if "threshold" in details:
                    min_val, max_val = map(float, details["threshold"].split("-"))
                    if not (min_val <= user_value <= max_val):
                        print(f"Error: {key} debe estar entre {min_val} y {max_val}. Intenta de nuevo.")
                        continue

                user_input[key] = user_value
                break  # Salir del loop si todo es correcto

            except ValueError:
                print(f"Error: {key} debe ser un {expected_type}. Intenta de nuevo.")

    return user_input


def get_predictions(api_url, input_data):
    """Envía los datos a la API del cliente y obtiene las predicciones"""
    response = requests.post(api_url, json=input_data)

    if response.status_code == 200:
        return response.json()  # Devuelve las predicciones como diccionario
    else:
        raise Exception(f"Error en la API: {response.status_code} - {response.text}")


def calculate_weighted_average(predictions, weights):
    """Calcula la media ponderada de las predicciones usando los pesos definidos"""
    total_weight = sum(weights.values())
    weighted_sum = sum(predictions[model] * weights[model] for model in weights)

    return weighted_sum / total_weight if total_weight > 0 else None


def main(yaml_path, api_url):
    """Carga el YAML, solicita datos al usuario, obtiene predicciones y devuelve la media ponderada"""

    # 1️⃣ Cargar el YAML
    syntax = load_syntax(yaml_path)

    # 2️⃣ Extraer los inputs comunes y modelos con pesos
    common_input = syntax.get("common_input", {})
    models = syntax.get("models", {})

    # 3️⃣ Pedir los datos al usuario y validarlos
    user_input = get_user_input(common_input)

    # 4️⃣ Extraer pesos de los modelos
    model_weights = {model: details["weight"] for model, details in models.items()}

    # 5️⃣ Obtener predicciones de la API
    predictions = get_predictions(api_url, user_input)

    # 6️⃣ Calcular la media ponderada
    weighted_avg = calculate_weighted_average(predictions, model_weights)

    # 7️⃣ Mostrar resultados
    print("\n🔹 **Predicciones individuales:**", predictions)
    print("🔹 **Media ponderada del coste:**", round(weighted_avg, 2), "€")

    return weighted_avg


if __name__ == "__main__":
    YAML_PATH = "sintaxis_common.yml"  # Ruta al YAML del cliente
    API_URL = "http://localhost:8000/predict"  # URL de la API del cliente  ESTO TENDRIA QUE OBTENERSE DEL YML
    main(YAML_PATH, API_URL)
