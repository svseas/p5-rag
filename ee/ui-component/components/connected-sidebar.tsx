"use client";

import { useMorphik } from "@/contexts/morphik-context";
import { MorphikSidebar } from "@/components/morphik-sidebar";

export function ConnectedSidebar() {
  const { connectionUri, updateConnectionUri, userProfile, onLogout, onProfileNavigate, onUpgradeClick } = useMorphik();

  return (
    <MorphikSidebar
      variant="inset"
      showEditableUri={true}
      connectionUri={connectionUri}
      onUriChange={updateConnectionUri}
      userProfile={userProfile}
      onLogout={onLogout}
      onProfileNavigate={onProfileNavigate}
      onUpgradeClick={onUpgradeClick}
    />
  );
}
