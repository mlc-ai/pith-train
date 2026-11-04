"""
Process-global runtime state, one module per concern, set once by its setup function so code reads
it in-line instead of threading it through every constructor and call.

The exceptions are weight_grad_store (the deferred weight gradients) and fp8_weight_cache (the
quantized-weight version counter), per-step state that DualPipeV.step drives. Import them
by class name.
"""
