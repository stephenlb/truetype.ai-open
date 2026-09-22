# Jev Open Source Blocks agent

This Python provider accepts the exact JSON body used by the local replica's
`POST /v1/systemone` endpoint in its `request` input part. It returns one
`response` artifact (`application/json`) containing the same `model`,
`answers`, and `usage` fields as that endpoint.

The handler executes the same Gemma inference implementation and honors the
same `TYPESAFE_REPLICA_*` environment settings as the Docker image. Its package
dependency installs the public replica repository; when run from this checkout,
the handler instead uses the sibling source tree so local changes take effect.

Install the agent dependencies:

```bash
cd agent && pip install -e .
blocks check
```

To use Blocks Network, authenticate and register it yourself:

```bash
cd agent
blocks login --network --write-env
blocks register
blocks run
```

Then run `python trigger.py` from `agent` to submit the example request.
