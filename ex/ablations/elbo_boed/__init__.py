"""elbo_boed: analytic-design (top-eigenvector) sequential BOED with an ELBO-
optimized alpha-tempered posterior. isolates the posterior channel: the design
rule is method-independent, so the only DRE-driven step is the posterior update.
reuses eig_elbo_boed's channel-agnostic core (study/design_select analytic branch/
posterior_opt/elbo_dre)."""
