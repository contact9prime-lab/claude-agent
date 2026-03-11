#!/usr/bin/env bash
#
# Deploy the Meeting Transcript Recorder to Google Cloud Functions.
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Authenticate: gcloud auth login
#   3. Set project: gcloud config set project YOUR_PROJECT_ID
#   4. Enable APIs:
#      gcloud services enable cloudfunctions.googleapis.com
#      gcloud services enable cloudbuild.googleapis.com
#      gcloud services enable cloudscheduler.googleapis.com
#      gcloud services enable firestore.googleapis.com
#      gcloud services enable storage.googleapis.com
#      gcloud services enable calendar-json.googleapis.com
#      gcloud services enable drive.googleapis.com
#
# Usage:
#   ./deploy.sh                    # Deploy both functions
#   ./deploy.sh process            # Deploy only the process function
#   ./deploy.sh poll               # Deploy only the poll function
#   ./deploy.sh scheduler          # Set up Cloud Scheduler for polling

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID environment variable}"
REGION="${GCP_REGION:-us-central1}"
BUCKET_NAME="${GCS_BUCKET_NAME:-meeting-transcripts-bucket}"

deploy_process_function() {
    echo "Deploying process_meeting_recording function..."
    gcloud functions deploy process_meeting_recording \
        --gen2 \
        --runtime python312 \
        --region "$REGION" \
        --source . \
        --entry-point process_meeting_recording \
        --trigger-http \
        --allow-unauthenticated=false \
        --memory 1Gi \
        --timeout 540s \
        --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,GCS_BUCKET_NAME=$BUCKET_NAME,OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --min-instances 0 \
        --max-instances 10
    echo "Done."
}

deploy_poll_function() {
    echo "Deploying poll_upcoming_meetings function..."
    gcloud functions deploy poll_upcoming_meetings \
        --gen2 \
        --runtime python312 \
        --region "$REGION" \
        --source . \
        --entry-point poll_upcoming_meetings \
        --trigger-http \
        --allow-unauthenticated=false \
        --memory 512Mi \
        --timeout 300s \
        --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,GCS_BUCKET_NAME=$BUCKET_NAME,OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --min-instances 0 \
        --max-instances 2
    echo "Done."
}

setup_scheduler() {
    echo "Setting up Cloud Scheduler to poll every 15 minutes..."
    POLL_URL=$(gcloud functions describe poll_upcoming_meetings \
        --gen2 --region "$REGION" --format="value(serviceConfig.uri)")

    gcloud scheduler jobs create http poll-meeting-recordings \
        --schedule "*/15 * * * *" \
        --uri "$POLL_URL" \
        --http-method POST \
        --oidc-service-account-email "${PROJECT_ID}@appspot.gserviceaccount.com" \
        --location "$REGION" \
        --description "Poll for new Google Meet recordings every 15 minutes" \
        --attempt-deadline 300s \
    || echo "Scheduler job may already exist. Use 'gcloud scheduler jobs update http poll-meeting-recordings ...' to update."

    echo "Done. Scheduler will invoke poll_upcoming_meetings every 15 minutes."
}

create_bucket() {
    echo "Ensuring GCS bucket exists: $BUCKET_NAME"
    gsutil ls -b "gs://$BUCKET_NAME" 2>/dev/null \
        || gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$BUCKET_NAME"
    echo "Done."
}

# --- Main ---
case "${1:-all}" in
    process)
        deploy_process_function
        ;;
    poll)
        deploy_poll_function
        ;;
    scheduler)
        setup_scheduler
        ;;
    bucket)
        create_bucket
        ;;
    all)
        create_bucket
        deploy_process_function
        deploy_poll_function
        echo ""
        echo "Both functions deployed. Run './deploy.sh scheduler' to set up automatic polling."
        ;;
    *)
        echo "Usage: $0 {all|process|poll|scheduler|bucket}"
        exit 1
        ;;
esac
