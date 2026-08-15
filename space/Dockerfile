# Clip explorer as a HuggingFace Docker Space.
#
# The app is public; the clips are not. At runtime it reads a PRIVATE dataset
# repo (HF_DATASET_REPO + HF_TOKEN) because AMASS-derived motion cannot be
# redistributed. With neither set the Space still boots and simply shows no
# releases.
#
# Built by .github/workflows/deploy-space.yml, which compiles the frontend and
# uploads this directory plus the app sources.

FROM node:20-slim AS frontend
WORKDIR /build
COPY clip_explorer/frontend/package.json clip_explorer/frontend/package-lock.json ./
RUN npm ci
COPY clip_explorer/frontend/ ./
RUN npm run build

FROM python:3.10-slim
# Spaces run as uid 1000; the HF cache must be writable for lazy video fetches
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    PYTHONPATH=/app
# ^ running `python clip_explorer/app.py` puts /app/clip_explorer on sys.path,
#   not /app, so the `retargeting.utils` shim needs PYTHONPATH to be found

WORKDIR /app
# only what the API needs — no torch, mujoco or smplx: the Space serves
# precomputed metadata and never retargets
RUN pip install --no-cache-dir flask "huggingface_hub>=0.34"

COPY --chown=user clip_explorer/app.py clip_explorer/hf_data.py ./clip_explorer/
COPY --chown=user retargeting/utils/clip_names.py ./retargeting/utils/clip_names.py
COPY --chown=user curation/test_split.json ./curation/test_split.json
COPY --from=frontend --chown=user /build/dist ./clip_explorer/frontend/dist

# the app imports `retargeting.utils.clip_names`; make the shim a real package
RUN touch retargeting/__init__.py retargeting/utils/__init__.py \
 && chown -R user /app

USER user
EXPOSE 7860
CMD ["python", "clip_explorer/app.py"]
