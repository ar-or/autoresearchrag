"""Test script to verify OpenTelemetry traces reach Langfuse."""

import time

from dotenv import load_dotenv

load_dotenv()

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

resource = Resource.create({"service.name": "gov_agentic"})
provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter()
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("gov_agentic.test")

with tracer.start_as_current_span("test-root-span") as root:
    root.set_attribute("test.type", "smoke_test")
    time.sleep(0.1)

    with tracer.start_as_current_span("child-step-1") as child1:
        child1.set_attribute("step", "fetch_data")
        time.sleep(0.05)

    with tracer.start_as_current_span("child-step-2") as child2:
        child2.set_attribute("step", "process_data")
        time.sleep(0.05)

provider.force_flush()
print(f"Trace ID: {format(root.get_span_context().trace_id, '032x')}")
print("Spans exported. Check Langfuse UI.")
