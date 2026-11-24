"""Qt-free helper for the app's own outbound HTTPS connections.

Every download the app performs itself (the CUDA-PyTorch wheels in
``vodparse.cli.gpu_support``, the GitHub Releases calls in ``vodparse.github_releases``)
opens its connection with the context from :func:`make_ssl_context` instead of
urllib's default.

Why: Python verifies TLS against a *snapshot* of the operating system's trust store.
On Windows that snapshot is whatever the certificate store holds right now; Python
cannot trigger Windows' on-demand root download the way Schannel clients (Edge,
PowerShell, most installers) do. A fresh Windows install carries neither the Amazon
nor the Starfield root that ``download.pytorch.org`` chains to until something
Schannel-based has visited an Amazon-signed site, so the GPU download failed with
``CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`` on a clean VM
and worked on one with browsing history (observed 2026-08-21). ``certifi`` ships
Mozilla's root bundle inside the app; loading it on top of the system roots makes
verification independent of that history, and keeping the system roots means a
corporate proxy's private root still works.
"""

from __future__ import annotations

import ssl


def make_ssl_context() -> ssl.SSLContext:
    """Return a verifying client ``SSLContext``: the system roots plus certifi's bundle.

    Starts from :func:`ssl.create_default_context` (certificate verification and hostname
    checking on, the system trust store loaded) and adds certifi's CA bundle to it.
    ``certifi`` is a declared dependency, but its ``cacert.pem`` data file could still be
    missing from a freeze; in that case the system-only context is returned rather than
    failing, which is exactly urllib's prior behavior.
    """
    context = ssl.create_default_context()
    try:
        import certifi
        context.load_verify_locations(cafile=certifi.where())
    except (ImportError, OSError):
        # No certifi, or its bundle is missing/unreadable: fall back to the system roots.
        pass
    return context
