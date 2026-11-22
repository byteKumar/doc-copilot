import AccessGate from "@/components/AccessGate";
import Chat from "@/components/Chat";

export default function Home() {
  return (
    <AccessGate>
      <Chat />
    </AccessGate>
  );
}
