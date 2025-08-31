import pytest
from unittest.mock import MagicMock, patch
from configparser import ConfigParser
import os
import json

from services.services import (
    ClientFactory,
    OpenAIClient,
    GoogleGenAIClient,
    GroqClient,
    MistralClient,
    AmazonBedrock,
)

@pytest.fixture
def temp_config(tmp_path, mocker):
    config_path = tmp_path / "zsh_codex.ini"
    mocker.patch("services.services.CONFIG_PATH", str(config_path))
    return config_path

def write_config(path, service_name, service_config):
    config = ConfigParser()
    config["service"] = {"service": service_name}
    config[service_name] = service_config
    with open(path, "w") as configfile:
        config.write(configfile)

def test_openai_client(temp_config, mocker):
    write_config(temp_config, "openai_test", {
        "api_type": "openai",
        "api_key": "test_key",
        "model": "test_model",
    })

    mock_openai_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "ls -la"
    mock_openai_client.chat.completions.create.return_value = mock_completion
    mocker.patch("openai.OpenAI", return_value=mock_openai_client)

    client = ClientFactory.create()
    assert isinstance(client, OpenAIClient)
    completion = client.get_completion("ls")

    assert completion == "ls -la"
    mock_openai_client.chat.completions.create.assert_called_once()

def test_google_genai_client(temp_config, mocker):
    write_config(temp_config, "google_test", {
        "api_type": "gemeni",
        "api_key": "test_key",
    })

    mock_genai_module = MagicMock()
    mock_model = MagicMock()
    mock_model.start_chat.return_value.send_message.return_value.text = "git status"
    mock_genai_module.GenerativeModel.return_value = mock_model
    mocker.patch.dict('sys.modules', {'google.generativeai': mock_genai_module})


    client = ClientFactory.create()
    assert isinstance(client, GoogleGenAIClient)
    completion = client.get_completion("git")

    assert completion == "git status"
    mock_genai_module.GenerativeModel.assert_called_once()
    mock_model.start_chat.return_value.send_message.assert_called_once()

def test_groq_client(temp_config, mocker):
    write_config(temp_config, "groq_test", {
        "api_type": "groq",
        "api_key": "test_key",
    })

    mock_groq_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "docker ps"
    mock_groq_client.chat.completions.create.return_value = mock_completion
    mocker.patch("groq.Groq", return_value=mock_groq_client)

    client = ClientFactory.create()
    assert isinstance(client, GroqClient)
    completion = client.get_completion("docker")

    assert completion == "docker ps"
    mock_groq_client.chat.completions.create.assert_called_once()

def test_mistral_client(temp_config, mocker):
    write_config(temp_config, "mistral_test", {
        "api_type": "mistral",
        "api_key": "test_key",
    })

    mock_mistral_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "kubectl get pods"
    mock_mistral_client.chat.complete.return_value = mock_completion
    mocker.patch("mistralai.Mistral", return_value=mock_mistral_client)

    client = ClientFactory.create()
    assert isinstance(client, MistralClient)
    completion = client.get_completion("kubectl")

    assert completion == "kubectl get pods"
    mock_mistral_client.chat.complete.assert_called_once()


def test_amazon_bedrock_client(temp_config, mocker):
    write_config(temp_config, "bedrock_test", {
        "api_type": "bedrock",
        "aws_access_key_id": "test_id",
        "aws_secret_access_key": "test_secret",
        "model": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    })

    mock_boto3_client = MagicMock()
    response_body = json.dumps({"content": [{"text": "aws s3 ls"}]})
    mock_response = {'body': MagicMock()}
    mock_response['body'].read.return_value = response_body.encode('utf-8')
    mock_boto3_client.invoke_model.return_value = mock_response
    mocker.patch("boto3.client", return_value=mock_boto3_client)

    client = ClientFactory.create()
    assert isinstance(client, AmazonBedrock)
    completion = client.get_completion("aws")

    assert completion == "aws s3 ls"
    mock_boto3_client.invoke_model.assert_called_once()


from create_completion import main as create_completion_main

def test_create_completion_main(mocker):
    # Mock sys.argv, sys.stdin, sys.stdout
    mocker.patch("sys.argv", ["create_completion.py", "5"])
    mock_stdin = MagicMock()
    mock_stdin.read.return_value = "echo "
    mocker.patch("sys.stdin", mock_stdin)
    mock_stdout = MagicMock()
    mocker.patch("sys.stdout", mock_stdout)

    # Mock ClientFactory
    mock_client = MagicMock()
    mock_client.get_completion.return_value = "echo hello"
    mocker.patch("services.services.ClientFactory.create", return_value=mock_client)

    # Run main
    create_completion_main()

    # Assertions
    mock_client.get_completion.assert_called_once()
    full_command_arg = mock_client.get_completion.call_args[0][0]
    assert full_command_arg == "#!/bin/zsh\n\necho "

    mock_stdout.write.assert_called_once_with("hello")
