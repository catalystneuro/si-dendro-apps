import dendro.sdk as pr

from .processor_pipeline import PipelineProcessor


app_name = 'si_kilosort25'

app = pr.App(
    name=app_name,
    help="Spike Interface Pipeline - Kilosort2.5",
    app_image=f"magland/{app_name}",
    app_executable="/app/main.py"
)


app.add_processor(PipelineProcessor)


if __name__ == '__main__':
    app.run()
