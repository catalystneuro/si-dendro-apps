FROM spikeinterface/kilosort3-compiled-base:0.2.0

# Install Dendro from source, for now
WORKDIR /src
RUN git clone https://github.com/flatironinstitute/dendro.git && \
    cd dendro/python && \
    pip install -e .

# Install spikeinterface-pipelines from source, for now
RUN git clone https://github.com/SpikeInterface/spikeinterface_pipelines.git && \
    cd spikeinterface_pipelines && \
    # git checkout dev && \
    pip install -e .

# Install neuroconv from source, for now
RUN git clone https://github.com/catalystneuro/neuroconv.git && \
    cd neuroconv && \
    pip install -e .

# # Install spikeinterface from source, for now
# RUN git clone https://github.com/SpikeInterface/spikeinterface.git && \
#     cd spikeinterface && \
#     # git checkout dev && \
#     pip install -e .[full]

# Install spikeinterface
RUN pip install spikeinterface[full]==0.100.0

# Copy files into the container
WORKDIR /app
COPY /common /app/common
COPY si_kilosort3/*.py /app/
COPY si_kilosort3/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Give execution permission to /app/main.py
RUN chmod +x /app/main.py
