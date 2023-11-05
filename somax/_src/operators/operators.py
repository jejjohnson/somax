from finitevolx import laplacian
import jax


laplacian_batch = jax.vmap(laplacian, in_axes=(0, None))