import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { api } from "../lib/api";
import type { SystemProblem } from "../lib/types";

// Surfaces Backend/API/services/system_check.py's findings - no usable GPU,
// or a model file a stage can't run without - before a video is uploaded,
// rather than as a failed job afterward. Renders nothing when everything
// checks out, or if the status can't be fetched at all (the rest of the UI
// already reports an unreachable backend). Dismissing only lasts until the
// page reloads, so a real problem keeps coming back until it's fixed.
export function SystemStatusBanner() {
  const [problems, setProblems] = useState<SystemProblem[]>([]);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    api
      .systemStatus()
      .then((status) => setProblems(status.problems.filter((p) => p.severity !== "info")))
      .catch(() => undefined);
  }, []);

  if (dismissed || problems.length === 0) return null;

  const hasError = problems.some((p) => p.severity === "error");

  return (
    <Alert severity={hasError ? "error" : "warning"} onClose={() => setDismissed(true)} sx={{ mb: 3 }}>
      <AlertTitle>
        {hasError ? "Processing new videos will fail until this is fixed" : "Not using the GPU"}
      </AlertTitle>
      {problems.map((problem) => (
        <Box key={problem.message} sx={{ mb: 1 }}>
          <Typography variant="body2">{problem.message}</Typography>
          <Typography
            variant="caption"
            component="div"
            sx={{ fontFamily: "monospace", opacity: 0.85, overflowWrap: "anywhere" }}
          >
            {problem.fix}
          </Typography>
        </Box>
      ))}
      <Typography variant="caption" component="div" sx={{ mt: 1 }}>
        Re-check with <code>python doctor.py</code> in Backend/, then restart the backend.
      </Typography>
    </Alert>
  );
}
