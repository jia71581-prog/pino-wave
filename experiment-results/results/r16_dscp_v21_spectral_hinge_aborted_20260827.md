# v21 spectral-hinge probe aborted before training

The initial v21 probe was stopped before any terminal or checkpoint was written. Its proposed training band matched the legacy DSCP `metric_record` last-axis thirds, whereas the paper's confirmatory metric uses normalized two-dimensional radial Fourier bands. Continuing would have optimized a mismatched target.

The frozen files are retained for provenance, but no result may be inferred from them. The replacement experiment is v20b, which first audits the frozen v18 B checkpoint under the exact confirmatory radial metric and tests the existing deployment-computable coefficient-space radial guard.
