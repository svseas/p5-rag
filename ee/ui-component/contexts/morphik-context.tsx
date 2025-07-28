"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { extractTokenFromUri, getApiBaseUrlFromUri } from "@/lib/utils";

const DEFAULT_API_BASE_URL = "http://localhost:8000";
const CONNECTION_URI_STORAGE_KEY = "morphik-connection-uri";

interface MorphikContextType {
  connectionUri: string | null;
  authToken: string | null;
  apiBaseUrl: string;
  isReadOnlyUri: boolean;
  updateConnectionUri: (uri: string) => void;
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;
  onUpgradeClick?: () => void;
  onBackClick?: () => void;
}

const MorphikContext = createContext<MorphikContextType | undefined>(undefined);

// Helper function to safely access localStorage
function getStoredConnectionUri(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(CONNECTION_URI_STORAGE_KEY);
  } catch {
    return null;
  }
}

function setStoredConnectionUri(uri: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (uri) {
      window.localStorage.setItem(CONNECTION_URI_STORAGE_KEY, uri);
    } else {
      window.localStorage.removeItem(CONNECTION_URI_STORAGE_KEY);
    }
  } catch {
    // Ignore localStorage errors
  }
}

export function MorphikProvider({
  children,
  initialConnectionUri = null,
  isReadOnlyUri = false,
  connectionUri: externalConnectionUri,
  onBackClick,
  userProfile,
  onLogout,
  onProfileNavigate,
  onUpgradeClick,
}: {
  children: React.ReactNode;
  initialConnectionUri?: string | null;
  isReadOnlyUri?: boolean;
  connectionUri?: string | null;
  onBackClick?: () => void;
  userProfile?: {
    name?: string;
    email?: string;
    avatar?: string;
    tier?: string;
  };
  onLogout?: () => void;
  onProfileNavigate?: (section: "account" | "billing" | "notifications") => void;
  onUpgradeClick?: () => void;
}) {
  const [connectionUri, setConnectionUri] = useState<string | null>(() => {
    // Priority: external prop > stored value > initial value
    return externalConnectionUri || getStoredConnectionUri() || initialConnectionUri;
  });

  const authToken = connectionUri ? extractTokenFromUri(connectionUri) : null;
  const apiBaseUrl = connectionUri ? getApiBaseUrlFromUri(connectionUri) : DEFAULT_API_BASE_URL;

  // Effect to persist connectionUri changes to localStorage
  useEffect(() => {
    setStoredConnectionUri(connectionUri);
  }, [connectionUri]);

  const updateConnectionUri = (uri: string) => {
    if (!isReadOnlyUri) {
      setConnectionUri(uri);
    }
  };

  return (
    <MorphikContext.Provider
      value={{
        connectionUri,
        authToken,
        apiBaseUrl,
        isReadOnlyUri,
        updateConnectionUri,
        userProfile,
        onLogout,
        onProfileNavigate,
        onUpgradeClick,
        onBackClick,
      }}
    >
      {children}
    </MorphikContext.Provider>
  );
}

export function useMorphik() {
  const context = useContext(MorphikContext);
  if (context === undefined) {
    throw new Error("useMorphik must be used within a MorphikProvider");
  }
  return context;
}
