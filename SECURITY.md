## Security

NVIDIA is dedicated to the security and trust of our software products and services, including all source code repositories managed through our organization.

If you need to report a security issue, please use the appropriate contact points outlined below. **Please do not report security vulnerabilities through GitHub.** If a potential security issue is inadvertently reported via a public issue or pull request, NVIDIA maintainers may limit public discussion and redirect the reporter to the appropriate private disclosure channels.

## Reporting Potential Security Vulnerability in an NVIDIA Product

To report a potential security vulnerability in any NVIDIA product:
- Web: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- E-Mail: psirt@nvidia.com
    - We encourage you to use the following PGP key for secure email communication: [NVIDIA public PGP Key for communication](https://www.nvidia.com/en-us/security/pgp-key)
    - Please include the following information:
   	 - Product/Driver name and version/branch that contains the vulnerability
     - Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
   	 - Instructions to reproduce the vulnerability
   	 - Proof-of-concept or exploit code
   	 - Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

While NVIDIA currently does not have a bug bounty program, we do offer acknowledgement when an externally reported security issue is addressed under our coordinated vulnerability disclosure policy. Please visit our [Product Security Incident Response Team (PSIRT)](https://www.nvidia.com/en-us/security/psirt-policies/) policies page for more information.

## NVIDIA Product Security

For all security-related concerns, please visit NVIDIA's Product Security portal at https://www.nvidia.com/en-us/security

## Trust boundary

**Every rank in a process group this library builds must belong to a single, mutually trusted job
admitted by one scheduler.** `fold_cp_ops` does not authenticate ranks, does not verify that a peer
is running the same code, and does not sandbox anything a peer sends. A process that can join the
group — or that can write into the NVSHMEM symmetric heap the group shares — is already inside the
boundary and is treated as trusted. Restricting who may join is the launcher's and the scheduler's
job, not this library's.

Two consequences worth stating plainly, because they are what the boundary buys:

- **Collective payloads are decoded, not executed.** The autotune config consensus
  (`fold_cp_ops/distributed/distributed_autotune.py`) broadcasts a MessagePack pack and decodes it
  with no object or extension hook, so a received buffer can construct primitives only. It used
  `pickle`, whose `loads` is arbitrary-code execution on whatever bytes arrive. Inside a trusted
  job that was not reachable by an attacker; the codec is narrowed anyway, because a defence that
  depends on the boundary holding everywhere is one that fails silently when it does not.
- **Persistent caches are private to the invoking user, and the WHOLE PATH is checked.** The JIT,
  artifact, autotune and distributed-autotune-freeze caches live under roots this library creates
  `0700` and validates before use. Validation walks **every component from `/` downward**, holding
  each parent's descriptor while opening or creating its child, so no component is resolved twice
  and no symlink anywhere in the chain can redirect the leaf. Checking only the final directory
  would not be enough: anyone able to rename a *parent* can substitute the entire subtree beneath
  it, leaving a `0700` leaf that is not private at all.

  An **ancestor** is accepted when it is a directory owned by the effective uid or by root and is
  not writable by others — with one exemption, for a **root-owned sticky** directory such as
  `/tmp`, where the sticky bit already restricts rename and removal to an entry's owner. The
  **final root** is held to more: effective-uid-owned and `0700`, with no sticky exemption, because
  that is the directory entries are read from and executed. Cache **files** must be regular,
  owner-owned and not writable by others; one that was already group- or world-writable is
  **refused rather than repaired**, since tightening it would only make untrusted content look
  private. Files are published `0600` via a private temporary and an atomic rename.

  The verdict for a given root is computed **once per process** and reused, so this costs one walk
  rather than one per compile, and a rejection warns exactly once. A rejected root **disables disk
  caching for that process and never blocks the run** — these are optimisations, and an outage is a
  worse outcome than a recompile.

  This closes cache poisoning by a *local* user outside the job, which the process-group trust
  boundary says nothing about, since such a user never joins the group at all.

- **A cache is not shareable across principals, and that is not a limitation to work around.**
  Every root must be owned by the user running the job. Pointing two users at one cache directory
  is unsupported: it necessarily makes each of them able to supply the other with an artifact that
  is loaded and executed. Where a team wants shared build output, share it through a channel that
  authenticates its contents — a signed artifact store — not through a directory both can write.

Neither mechanism is a substitute for the boundary. Do not run ranks from mutually distrusting
users in one group, and do not point a cache root at a shared, writable location.
