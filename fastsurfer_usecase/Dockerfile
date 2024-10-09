FROM inesgp/torch_nan:v0.2 as builder

FROM deepmi/fastsurfer:cpu-v2.1.1

RUN /bin/bash -c "mkdir -p /workspace"

COPY --from=builder /workspace/* /workspace/.
COPY --from=builder /opt/conda/ /opt/conda/

RUN cp /venv/lib/libstdc++.so.6 /opt/conda/lib/./libstdc++.so.6

RUN /bin/bash -c "source /venv/bin/activate && pip install /workspace/*.whl && rm /workspace/*"

WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]
