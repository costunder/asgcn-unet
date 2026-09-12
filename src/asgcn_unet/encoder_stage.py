"""Non-owning views of the SAME trained spline layers across a pooling boundary."""

from .graph import ASGCNEncoder


class EncoderStage:
    def __init__(self, encoder, start, stop, *, input_is_spiking=False):
        if not isinstance(encoder, ASGCNEncoder) or not 0 <= start < stop <= len(encoder.layers):
            raise ValueError("A stage must select a nonempty contiguous range of the ASGCN encoder")
        self.encoder = encoder
        self.layers = tuple(encoder.layers[start:stop])
        self.spline_backend = encoder.spline_backend
        self.input_is_spiking = input_is_spiking

    @property
    def training(self):
        return self.encoder.training

    def _basis_cache(self, graph):
        return ASGCNEncoder._basis_cache(self, graph)

    def forward_ann(self, graph, return_activations=False):
        return ASGCNEncoder.forward_ann(self, graph, return_activations)
