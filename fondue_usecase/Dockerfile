FROM inesgp/torch_nan:v0.2-noskip as build

FROM python:3.8-bullseye

RUN apt-get update

COPY --from=build /workspace/torch-2.4.0a0+git2dafa64-cp38-cp38-linux_x86_64.whl /wheels/torch-2.4.0a0+git2dafa64-cp38-cp38-linux_x86_64.whl
COPY --from=build /opt/conda /opt/conda
RUN /bin/bash -c "pip install /wheels/*.whl && rm /wheels/*.whl"

# RUN conda install albumentations
RUN pip install albumentations==1.3.0
RUN pip install nibabel==5.1.0
RUN pip install SimpleITK==2.2.1
RUN pip install h5py==3.7.0
RUN pip install imageio==2.19.3
#RUN pip install lpips==0.1.4
RUN pip install matplotlib==3.5.2
RUN pip install nibabel==5.1.0
#RUN pip install numpy==1.21.5
RUN pip install opencv_python==4.7.0.72
RUN pip install pandas==1.4.4
RUN pip install Pillow==9.5.0
#RUN pip install PYyaml==6.0
RUN pip install scikit_image==0.19.2
RUN pip install scikit_learn==1.0.2
RUN pip install scipy==1.9.1
RUN pip install tqdm==4.64.1
RUN pip install XlsxWriter==3.0.3
RUN apt install nano

ENTRYPOINT ["/bin/bash"]
