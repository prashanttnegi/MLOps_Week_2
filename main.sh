! pip install -r requirements.txt

! pip3 install --upgrade --quiet  google-cloud-aiplatform

! gsutil mb -l "us-central1" -p "sharp-cursor-461116-t1" "gs://mlops-course-sharp-cursor-461116-t1-week2"

echo "Created bucket gs://mlops-course-sharp-cursor-461116-t1-week2"

echo "Making model artifacts directory"

mkdir -p artifacts

python main.py

echo "Uploading model artifacts to gs://mlops-course-sharp-cursor-461116-t1-week2/my-models/iris-classifier-week-2/"

! gsutil cp artifacts/model.joblib "gs://mlops-course-sharp-cursor-461116-t1-week2"/"my-models/iris-classifier-week-2"/