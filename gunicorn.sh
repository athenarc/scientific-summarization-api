gunicorn -w 4 -k uvicorn.workers.UvicornWorker summarizer_api:app \
--bind 0.0.0.0:8000 \
--daemon \
--access-logfile /var/log/summarizer_api_access.log \
--error-logfile /var/log/summarizer_api_error.log \
--log-level info