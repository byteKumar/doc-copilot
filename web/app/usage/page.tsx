import type { Metadata } from "next";
import Usage from "@/components/Usage";

export const metadata: Metadata = { title: "Usage · doc-copilot" };

export default function UsagePage() {
  return <Usage />;
}
