import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "doc-copilot",
  description: "Answers from SEC 10-K filings, with citations, or an honest \"I don't know\".",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <Link href="/" className="brand">doc-copilot</Link>
          <span className="tagline">10-K answers with citations</span>
          <nav>
            <Link href="/">Chat</Link>
            <Link href="/usage">Usage</Link>
          </nav>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
