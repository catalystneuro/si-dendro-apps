from dendro.sdk import App

from processor_pipeline import PipelineProcessor


app_name = 'si_kilosort25'

app = App(
    name=app_name,
    description="Spike Interface Pipeline - Kilosort2.5",
    app_image=f"ghcr.io/catalystneuro/{app_name}",
    app_executable="/app/main.py"
)


app.add_processor(PipelineProcessor)


if __name__ == '__main__':
    app.run()
