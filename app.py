import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image

MODEL_DIR = Path("model")
KERNEL_SLUG = "olaelshiekh/cnn-realdata-covid19"
MODEL_FILENAME = "best_model.keras"
CLASS_NAMES_FILENAME = "class_names.json"


def _clean_secret_value(value: Any) -> str:
	"""Normalize secret values from env/secrets, including accidental wrapping quotes/brackets."""
	if value is None:
		return ""
	cleaned = str(value).strip()
	if (cleaned.startswith('"') and cleaned.endswith('"')) or (
		cleaned.startswith("'") and cleaned.endswith("'")
	):
		cleaned = cleaned[1:-1].strip()
	if (cleaned.startswith("<") and cleaned.endswith(">")):
		cleaned = cleaned[1:-1].strip()
	return cleaned


def _get_secret(name: str) -> str:
	# Priority: environment variable, then Streamlit secrets (flat and [kaggle] section).
	env_value = _clean_secret_value(os.getenv(name))
	if env_value:
		return env_value

	try:
		if name in st.secrets:
			return _clean_secret_value(st.secrets[name])
		if "kaggle" in st.secrets and name in st.secrets["kaggle"]:
			return _clean_secret_value(st.secrets["kaggle"][name])
	except Exception:
		# Accessing st.secrets can fail in some local contexts; fallback to env-only behavior.
		pass

	return ""


def _configure_kaggle_auth() -> None:
	load_dotenv(dotenv_path=Path(".env"), override=False)

	username = _get_secret("KAGGLE_USERNAME")
	kaggle_key = _get_secret("KAGGLE_KEY")
	api_token = _get_secret("KAGGLE_API_TOKEN")

	if not api_token and not (username and kaggle_key):
		raise RuntimeError(
			"Missing Kaggle auth. Configure KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY, in Streamlit secrets or .env."
		)

	kaggle_dir = Path.home() / ".kaggle"
	kaggle_dir.mkdir(parents=True, exist_ok=True)

	if api_token:
		os.environ["KAGGLE_API_TOKEN"] = api_token
		token_file = kaggle_dir / "access_token"
		token_file.write_text(api_token, encoding="utf-8")
		os.chmod(token_file, 0o600)

	if username and kaggle_key:
		os.environ["KAGGLE_USERNAME"] = username
		os.environ["KAGGLE_KEY"] = kaggle_key
		legacy_file = kaggle_dir / "kaggle.json"
		legacy_file.write_text(json.dumps({"username": username, "key": kaggle_key}), encoding="utf-8")
		os.chmod(legacy_file, 0o600)


def _run_kaggle_cli(args: list[str]) -> None:
	command = ["kaggle", *args]
	result = subprocess.run(command, capture_output=True, text=True)
	if result.returncode != 0:
		stderr = (result.stderr or "").strip()
		stdout = (result.stdout or "").strip()
		details = stderr or stdout or "Unknown Kaggle CLI error."
		if "authenticate" in details.lower() or "unauthorized" in details.lower():
			details = (
				"Kaggle authentication failed. Verify Streamlit secrets/.env values for "
				"KAGGLE_API_TOKEN or KAGGLE_USERNAME + KAGGLE_KEY."
			)
		raise RuntimeError(details)


def _find_named_file(file_name: str) -> Path | None:
	if not MODEL_DIR.exists():
		return None

	files = sorted(MODEL_DIR.rglob(file_name))
	if files:
		return files[0]
	return None


def _download_notebook_artifacts() -> tuple[Path, Path]:
	MODEL_DIR.mkdir(parents=True, exist_ok=True)

	model_file = _find_named_file(MODEL_FILENAME)
	class_names_file = _find_named_file(CLASS_NAMES_FILENAME)
	if model_file and class_names_file:
		return model_file, class_names_file

	_configure_kaggle_auth()

	_run_kaggle_cli(["kernels", "output", KERNEL_SLUG, "-p", str(MODEL_DIR)])

	model_file = _find_named_file(MODEL_FILENAME)
	class_names_file = _find_named_file(CLASS_NAMES_FILENAME)
	if not model_file or not class_names_file:
		raise RuntimeError(
			f"Notebook output must contain both {MODEL_FILENAME} and {CLASS_NAMES_FILENAME}."
		)

	return model_file, class_names_file


def _load_class_names(class_names_path: Path) -> list[str]:
	content: Any = json.loads(class_names_path.read_text(encoding="utf-8"))
	if not isinstance(content, list) or not content:
		raise RuntimeError("class_names.json must be a non-empty JSON array of class labels.")

	class_names = [str(name).strip() for name in content]
	if any(not name for name in class_names):
		raise RuntimeError("class_names.json includes empty class labels.")
	return class_names


@st.cache_resource
def load_model_and_classes() -> tuple[tf.keras.Model, list[str], Path, Path]:
	model_path, class_names_path = _download_notebook_artifacts()
	model = tf.keras.models.load_model(model_path)
	class_names = _load_class_names(class_names_path)
	return model, class_names, model_path, class_names_path


def _prepare_image(image: Image.Image, model: tf.keras.Model) -> np.ndarray:
	input_shape = model.input_shape
	if isinstance(input_shape, list):
		input_shape = input_shape[0]

	if len(input_shape) != 4:
		raise RuntimeError("Unexpected model input shape. Expected (None, H, W, C).")

	_, height, width, channels = input_shape
	if channels == 1:
		image = image.convert("L")
	else:
		image = image.convert("RGB")

	image = image.resize((width, height))
	image_array = np.array(image, dtype=np.float32) / 255.0

	if channels == 1:
		image_array = np.expand_dims(image_array, axis=-1)

	return np.expand_dims(image_array, axis=0)


def _predict(image: Image.Image, model: tf.keras.Model, class_names: list[str]) -> tuple[str, float]:
	input_tensor = _prepare_image(image, model)
	prediction = model.predict(input_tensor, verbose=0)
	scores = prediction[0]

	if np.ndim(scores) == 0:
		scores = np.array([scores], dtype=np.float32)

	if scores.shape[0] != len(class_names):
		raise RuntimeError(
			f"Model output has {scores.shape[0]} classes, expected {len(class_names)} from class_names.json."
		)

	if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(np.sum(scores), 1.0, atol=1e-3):
		scores = tf.nn.softmax(scores).numpy()

	index = int(np.argmax(scores))
	label = class_names[index]
	confidence = float(scores[index])
	return label, confidence


st.set_page_config(page_title="COVID X-ray Diagnosis", page_icon="🩻", layout="centered")
st.title("X-ray Disease Diagnosis")
st.write("Upload one chest X-ray image and diagnose it using the Kaggle notebook output model.")

try:
	model, class_names, model_path, class_names_path = load_model_and_classes()
except Exception as load_error:
	st.error(f"Model initialization failed: {load_error}")
	st.stop()

st.caption(f"Using model: {model_path.name}")
st.caption(f"Using classes: {class_names_path.name} → {class_names}")

uploaded_image = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
	image = Image.open(uploaded_image)
	st.image(image, caption="Uploaded X-ray", use_container_width=True)

	if st.button("Diagnose"):
		try:
			predicted_label, confidence_score = _predict(image, model, class_names)
			st.success(f"Prediction: {predicted_label}")
			st.info(f"Confidence: {confidence_score * 100:.2f}%")
		except Exception as predict_error:
			st.error(f"Prediction failed: {predict_error}")
