import type { Metadata } from "next";
import { cookies } from "next/headers";
import localFont from "next/font/local";
import { NextIntlClientProvider } from "next-intl";
import { getMessages } from "next-intl/server";
import "../globals.css";
import { AlertSystem } from "@/components/ui/alert-system";
import { ThemeProvider } from "@/components/theme-provider";
import { DynamicSiteHeader } from "@/components/dynamic-site-header";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar-components";
import { MorphikProvider } from "@/contexts/morphik-context";
import { HeaderProvider } from "@/contexts/header-context";
import { ChatProvider } from "@/components/chat/chat-context";
import { SidebarContainer } from "@/components/sidebar-container";

const geistSans = localFont({
  src: "../fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "../fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Morphik Dashboard",
  description: "Morphik - Knowledge Graph and RAG Platform",
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  // Server-side read of sidebar cookie to avoid SSR flicker
  const sidebarCookie = cookies().get("sidebar_state")?.value;
  const defaultOpen = sidebarCookie === undefined ? true : sidebarCookie === "true";

  // Get messages for next-intl
  const messages = await getMessages();

  return (
    <html lang="vi" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <NextIntlClientProvider messages={messages}>
          <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
            <div className="min-h-screen bg-sidebar">
              <MorphikProvider>
                <HeaderProvider>
                  <ChatProvider>
                    <SidebarProvider
                      defaultOpen={defaultOpen}
                      style={
                        {
                          "--sidebar-width": "calc(var(--spacing) * 72)",
                          "--header-height": "calc(var(--spacing) * 12)",
                        } as React.CSSProperties
                      }
                    >
                      <SidebarContainer />
                      <SidebarInset>
                        <DynamicSiteHeader />
                        <div className="flex flex-1 flex-col p-4 md:p-6">{children}</div>
                      </SidebarInset>
                    </SidebarProvider>
                  </ChatProvider>
                </HeaderProvider>
              </MorphikProvider>
            </div>
            <AlertSystem position="bottom-right" />
          </ThemeProvider>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
