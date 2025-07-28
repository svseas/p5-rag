/*
 * MorphikSidebar - LOCAL DEVELOPMENT VERSION
 *
 * This sidebar is used by the local ui-component dev server via:
 * layout.tsx → ConnectedSidebar → MorphikSidebar
 *
 * Features:
 * - URL-based navigation (traditional Next.js routing)
 * - URI editing capability for local development
 * - localStorage persistence for connection settings
 * - Used for standalone development and testing
 *
 * Safe to modify for local dev features - does NOT affect cloud UI!
 * The cloud UI uses morphik-sidebar-stateful.tsx instead.
 */
"use client";

import * as React from "react";
import Image from "next/image";
import {
  IconFiles,
  IconSearch,
  IconMessage,
  IconShare,
  IconPlugConnected,
  IconFileText,
  IconSettings,
  IconFileAnalytics,
  IconGitBranch,
  IconBook,
  IconMessageCircle,
  IconArrowRight,
  IconLink,
} from "@tabler/icons-react";

import { NavMain } from "@/components/nav-main";
import { NavSecondary } from "@/components/nav-secondary";
import { NavUser } from "@/components/nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarGroup,
  SidebarGroupContent,
} from "@/components/ui/sidebar-new";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { normalizeToMorphikUri, getApiBaseUrlFromUri } from "@/lib/utils";

const data = {
  user: {
    name: "Morphik User",
    email: "user@morphik.ai",
    avatar: "/assets/placeholder-user.jpg",
  },
  navMain: [
    {
      title: "Documents",
      url: "/documents",
      icon: IconFiles,
    },
    {
      title: "PDF Viewer",
      url: "/pdf",
      icon: IconFileText,
    },
    {
      title: "Search",
      url: "/search",
      icon: IconSearch,
    },
    {
      title: "Chat",
      url: "/chat",
      icon: IconMessage,
    },
    {
      title: "Knowledge Graph",
      url: "/graphs",
      icon: IconShare,
    },
    {
      title: "Workflows",
      url: "/workflows",
      icon: IconGitBranch,
    },
    {
      title: "Connections",
      url: "/connections",
      icon: IconPlugConnected,
    },
  ],
  navSecondary: [
    {
      title: "Settings",
      url: "/settings",
      icon: IconSettings,
    },
    {
      title: "Logs",
      url: "/logs",
      icon: IconFileAnalytics,
    },
    {
      title: "Documentation",
      url: "https://docs.morphik.ai",
      icon: IconBook,
    },
    {
      title: "Send Feedback",
      url: "mailto:founders@morphik.ai",
      icon: IconMessageCircle,
    },
  ],
};

interface MorphikSidebarProps extends React.ComponentProps<typeof Sidebar> {
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;
  onUpgradeClick?: () => void;
  showEditableUri?: boolean;
  connectionUri?: string | null;
  onUriChange?: (newUri: string) => void;
}

export function MorphikSidebar({
  userProfile,
  onLogout,
  onProfileNavigate,
  onUpgradeClick,
  showEditableUri = true,
  connectionUri,
  onUriChange,
  ...props
}: MorphikSidebarProps) {
  const [mounted, setMounted] = React.useState(false);
  const [uriInput, setUriInput] = React.useState(connectionUri || "");

  // Ensure component is mounted before rendering theme-dependent content
  React.useEffect(() => {
    setMounted(true);
  }, []);

  React.useEffect(() => {
    setUriInput(connectionUri || "");
  }, [connectionUri]);

  const handleUriSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onUriChange && uriInput.trim()) {
      const normalizedUri = normalizeToMorphikUri(uriInput.trim());
      onUriChange(normalizedUri);
    }
  };

  // Display the current API URL for user feedback
  const currentApiUrl = connectionUri ? getApiBaseUrlFromUri(connectionUri) : "Not connected";

  // Merge user profile with defaults
  const userData = {
    name: userProfile?.name || data.user.name,
    email: userProfile?.email || data.user.email,
    avatar: userProfile?.avatar || data.user.avatar,
  };

  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild className="data-[slot=sidebar-menu-button]:!p-2">
              <a href="/" className="flex items-center">
                {mounted && (
                  <>
                    <Image
                      src="/morphikwhite.png"
                      alt="Morphik Logo"
                      width={120}
                      height={32}
                      className="hidden h-8 w-auto object-contain dark:block"
                      priority
                    />
                    <Image
                      src="/morphikblack.png"
                      alt="Morphik Logo"
                      width={120}
                      height={32}
                      className="block h-8 w-auto object-contain dark:hidden"
                      priority
                    />
                  </>
                )}
                {!mounted && <div className="h-8 w-[120px]" />}
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        {showEditableUri && (
          <SidebarGroup>
            <SidebarGroupContent className="px-2 py-2">
              <form onSubmit={handleUriSubmit} className="space-y-2">
                <Label htmlFor="connection-uri" className="text-xs text-muted-foreground">
                  Connection URI
                </Label>
                <div className="flex gap-2">
                  <Input
                    id="connection-uri"
                    type="text"
                    placeholder="http://localhost:8000"
                    value={uriInput}
                    onChange={e => setUriInput(e.target.value)}
                    className="text-xs"
                  />
                  <Button type="submit" size="sm" variant="outline">
                    <IconLink className="h-3 w-3" />
                  </Button>
                </div>
                {mounted && <div className="text-xs text-muted-foreground">Current: {currentApiUrl}</div>}
              </form>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
        <NavMain items={data.navMain} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
        {/* Show upgrade button for free users in cloud UI only */}
        {onUpgradeClick && (userProfile?.tier === "free" || !userProfile?.tier) && (
          <div className="mx-2 mb-2 mt-2">
            <Button className="w-full justify-between" variant="outline" size="default" onClick={onUpgradeClick}>
              <div className="flex items-center gap-2">
                <span>Upgrade to</span>
                <Badge variant="secondary" className="text-xs">
                  PRO
                </Badge>
              </div>
              <IconArrowRight className="h-4 w-4" />
            </Button>
          </div>
        )}
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={userData} onLogout={onLogout} onProfileNavigate={onProfileNavigate} />
      </SidebarFooter>
    </Sidebar>
  );
}
