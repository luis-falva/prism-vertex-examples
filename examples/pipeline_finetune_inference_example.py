from prism_vertex.pipelines import factory
from prism_vertex.vertex.deploy import DeployPipeline


def finetune_pipeline(
        bucket_name: str = "pocv29-66dd-prism_vertex-ai",
        test_features_file: str = "sample_data_100.csv",
        test_batch_size: int = 1024,
        split: str = "test"
):
    # IMPORTANT NOTE: We can separate in Parallel and  Non parallel tasks.
    # For parallel processes just remove .after() method and leave it as it is.
    # For non parallel and chained processes this will require the usage of .after() method.

    # Step 1: Create DataLoader, this will write a .pt file from a TensorDataset
    create_dataloader_task = create_dataloader_op(
        f"gs://{bucket_name}/{test_features_file}"
    )

    # Step 2: Finetune model prediction, this will load the .pt file from a TensorDataset into a
    # DataLoader object and process as torch model type
    finetune_model_predict_task = finetune_model_predict_op(
        bucket_name,
        split,
        test_batch_size
    ).after(create_dataloader_task)

    # Step 3: Upload data into GCS, this will a predictions test csv file with probs output into pocv29
    upload_data_to_gcs_op(
        finetune_model_predict_task.outputs["artifact"]
    ).after(finetune_model_predict_task)


if __name__ == '__main__':

    job = DeployPipeline(
        func=finetune_pipeline,
        name="finetune-vertex",
        description="Bert Finetune Model Pipeline",
        pipeline_root=PIPELINE_ROOT,
        template_path="pipeline_finetune_inference"
    )
    job.submit()
