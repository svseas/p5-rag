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
} from "@tabler/icons-react";

export interface BaseNavItem {
  title: string;
  icon: React.ComponentType;
  isSpecial?: boolean;
}

export interface UrlNavItem extends BaseNavItem {
  type: "url";
  url: string;
}

export interface SectionNavItem extends BaseNavItem {
  type: "section";
  section: string;
}

export type NavItem = UrlNavItem | SectionNavItem;

export interface NavigationStrategy {
  type: "url" | "section";
  mainItems: NavItem[];
  secondaryItems: NavItem[];
  onItemClick: (item: NavItem) => void;
  currentActive?: string;
}

// Navigation keys for i18n - actual titles will be translated at render time
export const baseMainNavItems: Omit<BaseNavItem, "type">[] = [
  {
    title: "nav.documents",
    icon: IconFiles,
  },
  {
    title: "nav.pdfViewer",
    icon: IconFileText,
  },
  {
    title: "nav.search",
    icon: IconSearch,
  },
  {
    title: "nav.chat",
    icon: IconMessage,
    isSpecial: true,
  },
  {
    title: "nav.knowledgeGraphs",
    icon: IconShare,
  },
  {
    title: "nav.workflows",
    icon: IconGitBranch,
  },
  {
    title: "nav.connectors",
    icon: IconPlugConnected,
  },
];

export const baseSecondaryNavItems: Omit<BaseNavItem, "type">[] = [
  {
    title: "nav.settings",
    icon: IconSettings,
    isSpecial: true,
  },
  {
    title: "nav.logs",
    icon: IconFileAnalytics,
  },
];

export const externalNavItems = [
  {
    title: "nav.documentation",
    url: "https://docs.morphik.ai",
    icon: IconBook,
  },
  {
    title: "nav.sendFeedback",
    url: "mailto:founders@morphik.ai",
    icon: IconMessageCircle,
  },
];

export const createUrlNavigation = (onChatClick: () => void, onSettingsClick?: () => void): NavigationStrategy => ({
  type: "url",
  mainItems: baseMainNavItems.map((item, index) => ({
    ...item,
    type: "url" as const,
    url:
      index === 0
        ? "/documents"
        : index === 1
          ? "/pdf"
          : index === 2
            ? "/search"
            : index === 3
              ? "/chat"
              : index === 4
                ? "/graphs"
                : index === 5
                  ? "/workflows"
                  : "/connections",
  })),
  secondaryItems: baseSecondaryNavItems.map((item, index) => ({
    ...item,
    type: "url" as const,
    url: index === 0 ? "/settings" : "/logs",
  })),
  onItemClick: item => {
    if ("url" in item) {
      // Settings: no dedicated page in local dev. Show overlay only.
      if (item.isSpecial && item.url === "/settings" && onSettingsClick) {
        // No dedicated /settings page in local dev. Show settings overlay only.
        onSettingsClick();
        return;
      }

      // Chat: if already on /chat, just open overlay. Otherwise navigate.
      if (item.isSpecial && item.url === "/chat") {
        if (typeof window !== "undefined" && window.location.pathname === "/chat") {
          onChatClick();
          return;
        }
      }

      window.location.href = item.url;
    }
  },
});

export const createSectionNavigation = (
  onSectionChange: (section: string) => void,
  onChatClick: () => void,
  currentSection?: string,
  onSettingsClick?: () => void
): NavigationStrategy => ({
  type: "section",
  mainItems: baseMainNavItems.map((item, index) => ({
    ...item,
    type: "section" as const,
    section:
      index === 0
        ? "documents"
        : index === 1
          ? "pdf"
          : index === 2
            ? "search"
            : index === 3
              ? "chat"
              : index === 4
                ? "graphs"
                : index === 5
                  ? "workflows"
                  : "connections",
  })),
  secondaryItems: baseSecondaryNavItems.map((item, index) => ({
    ...item,
    type: "section" as const,
    section: index === 0 ? "settings" : "logs",
  })),
  currentActive: currentSection,
  onItemClick: item => {
    if (item.isSpecial && "section" in item && item.section === "chat") {
      onChatClick();
      onSectionChange(item.section);
    } else if (item.isSpecial && "section" in item && item.section === "settings" && onSettingsClick) {
      onSettingsClick();
      onSectionChange(item.section);
    } else if ("section" in item) {
      onSectionChange(item.section);
    }
  },
});
