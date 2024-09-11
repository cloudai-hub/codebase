import sagemaker
import boto3
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve

# Initialize SageMaker session
sess = sagemaker.Session()
region = boto3.Session().region_name

# Retrieve the linear learner container
container = retrieve('linear-learner', region=region)

# Define SageMaker Estimator for Linear Learner
linear_estimator = Estimator(
    image_uri=container,
    role=get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=f's3://{bucket}/{prefix}/output',
    sagemaker_session=sess
)

# Set hyperparameters for Linear Learner
linear_estimator.set_hyperparameters(
    predictor_type='regressor',  # For linear regression
    mini_batch_size=100,
    epochs=100,
    learning_rate=0.01,
    normalize_data=True
)

# Define data channels for training and validation
train_input = TrainingInput(s3_data=s3_input_train, content_type='text/csv')
validation_input = TrainingInput(s3_data=s3_input_validation, content_type='text/csv')

# Fit the model with training data
linear_estimator.fit({'train': train_input, 'validation': validation_input})






# Features and target variable
X = data[['rooms', 'sqft']]
y = data['price']

# Normalize the input features to range 0 to 1
X_normalized = (X - X.min()) / (X.max() - X.min())

# Combine the normalized features with the target variable and make 'price' the first column
normalized_data = pd.concat([y, X_normalized], axis=1)

# Shuffle the dataset
normalized_data = normalized_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into training (80%) and validation (20%) sets
train_size = int(0.8 * len(normalized_data))
train_df = normalized_data[:train_size]
val_df = normalized_data[train_size:]











train_path = f"s3://{bucket}/{prefix}/train"
validation_path = f"s3://{bucket}/{prefix}/validation"
test_path = f"s3://{bucket}/{prefix}/test"

container = sagemaker.image_uris.retrieve(region=boto3.Session().region_name, framework='xgboost', version='latest')

s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')

sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role, 
                                    instance_count=1, 
                                    instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 

xgb_predictor = xgb.deploy(initial_instance_count=1,
                           instance_type='ml.m4.xlarge')



import boto3

client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")

model_artifacts = xgb.model_data
model_artifacts


from time import gmtime, strftime

model_name = "xgboost-serverless" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Model name: " + model_name)

# dummy environment variables
byo_container_env_vars = {"SAGEMAKER_CONTAINER_LOG_LEVEL": "20", "SOME_ENV_VAR": "myEnvVar"}

create_model_response = client.create_model(
    ModelName=model_name,
    Containers=[
        {
            "Image": container,
            "Mode": "SingleModel",
            "ModelDataUrl": model_artifacts,
            "Environment": byo_container_env_vars,
        }
    ],
    ExecutionRoleArn=role,
)

print("Model Arn: " + create_model_response["ModelArn"])
xgboost_epc_name = "mlops-serverless-epc" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

endpoint_config_response = client.create_endpoint_config(
    EndpointConfigName=xgboost_epc_name,
    ProductionVariants=[
        {
            "VariantName": "byoVariant",
            "ModelName": model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": 4096,
                "MaxConcurrency": 1,
            },
        },
    ],
)

print("Endpoint Configuration Arn: " + endpoint_config_response["EndpointConfigArn"])
endpoint_name = "xgboost-serverless-ep" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

create_endpoint_response = client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=xgboost_epc_name,
)

print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])
