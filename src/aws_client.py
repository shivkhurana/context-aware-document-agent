import boto3
import os
from botocore.exceptions import ClientError
from typing import Optional


class S3Client:
    def __init__(self):
        self.bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    def upload_text_document(self, filename: str, content: str) -> bool:
        """Upload a text document to S3 bucket."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=content.encode("utf-8"),
                ContentType="text/plain",
            )
            return True
        except ClientError as e:
            print(f"Error uploading {filename} to S3: {e}")
            return False

    def download_text_document(self, filename: str) -> Optional[str]:
        """Download a text document from S3 bucket."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
            content = response["Body"].read().decode("utf-8")
            return content
        except ClientError as e:
            print(f"Error downloading {filename} from S3: {e}")
            return None

    def delete_document(self, filename: str) -> bool:
        """Delete a document from S3 bucket."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
            return True
        except ClientError as e:
            print(f"Error deleting {filename} from S3: {e}")
            return False


# Global instance
s3_client = S3Client()
