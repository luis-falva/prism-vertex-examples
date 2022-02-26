from kfp.v2.dsl import Artifact


def disease_predictor(
        input_path: str,
        output_path: str,
        model_path: str,
        artifacts_path: str,
        dataloader_chunk_size: int,
        cpu_input_max_rows: int,
        token_chunk_size: int
) -> Artifact:
    """
    Important to notice that, inner imports i.e `from prism_ai.lumiata.prism...`
    refers to the prism-ai image at artifact registry docker repository.
    """
    import logging
    from prism_ai.lumiata.predict.disease.disease_predictor_manager import (
        DiseasePredictorManager,
        DiseasePredictorArguments,
    )
    logging.getLogger().setLevel(logging.INFO)

    disease_args = DiseasePredictorArguments(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        artifacts_path=artifacts_path,
        dataloader_chunk_size=dataloader_chunk_size,
        cpu_input_max_rows=cpu_input_max_rows,
        token_chunk_size=token_chunk_size,
    )

    with DiseasePredictorManager(disease_args) as predictor:
        predictor.predict()

    return disease_args
