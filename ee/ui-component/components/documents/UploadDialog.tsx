"use client";

import React, { useState, ChangeEvent } from "react";
import { useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
// Alert system is handled by the parent component

interface UploadDialogProps {
  showUploadDialog: boolean;
  setShowUploadDialog: (show: boolean) => void;
  loading: boolean;
  onFileUpload: (file: File | null, metadata: string, rules: string, useColpali: boolean) => Promise<void>;
  onBatchFileUpload: (files: File[], metadata: string, rules: string, useColpali: boolean) => Promise<void>;
  onTextUpload: (text: string, metadata: string, rules: string, useColpali: boolean) => Promise<void>;
}

const UploadDialog: React.FC<UploadDialogProps> = ({
  showUploadDialog,
  setShowUploadDialog,
  loading,
  onFileUpload,
  onBatchFileUpload,
  onTextUpload,
}) => {
  const t = useTranslations();

  // Component state for managing the upload form
  const [uploadType, setUploadType] = useState<"file" | "text" | "batch">("file");
  const [textContent, setTextContent] = useState("");
  // Used in handleFileChange and for providing to the parent component
  const [fileToUpload, setFileToUpload] = useState<File | null>(null);
  const [batchFilesToUpload, setBatchFilesToUpload] = useState<File[]>([]);
  const [metadata, setMetadata] = useState("{}");
  const [rules, setRules] = useState("[]");
  const [useColpali, setUseColpali] = useState(true);

  // Reset upload dialog state
  const resetUploadDialog = () => {
    setUploadType("file");
    setFileToUpload(null);
    setBatchFilesToUpload([]);
    setTextContent("");
    setMetadata("{}");
    setRules("[]");
    setUseColpali(true);
  };

  // Handle file selection
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setFileToUpload(files[0]);
    }
  };

  // Handle batch file selection
  const handleBatchFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setBatchFilesToUpload(Array.from(files));
    }
  };

  /*
   * Component state flow:
   * - Internal state is managed here (uploadType, fileToUpload, etc.)
   * - Actions like file upload are handled by parent component via callbacks
   * - No need to expose getter/setter methods as the parent has its own state
   */

  return (
    <Dialog
      open={showUploadDialog}
      onOpenChange={open => {
        setShowUploadDialog(open);
        if (!open) resetUploadDialog();
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("documents.uploadDocument")}</DialogTitle>
          <DialogDescription>{t("documents.uploadDocumentDescription")}</DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-4">
          <div className="flex gap-2">
            <Button variant={uploadType === "file" ? "default" : "outline"} onClick={() => setUploadType("file")}>
              {t("documents.file")}
            </Button>
            <Button variant={uploadType === "batch" ? "default" : "outline"} onClick={() => setUploadType("batch")}>
              {t("documents.batchFiles")}
            </Button>
            <Button variant={uploadType === "text" ? "default" : "outline"} onClick={() => setUploadType("text")}>
              {t("documents.text")}
            </Button>
          </div>

          {uploadType === "file" ? (
            <div>
              <Label htmlFor="file" className="mb-2 block">
                {t("documents.file")}
              </Label>
              <Input id="file" type="file" onChange={handleFileChange} />
            </div>
          ) : uploadType === "batch" ? (
            <div>
              <Label htmlFor="batchFiles" className="mb-2 block">
                {t("documents.selectFiles")}
              </Label>
              <Input id="batchFiles" type="file" multiple onChange={handleBatchFileChange} />
              {batchFilesToUpload.length > 0 && (
                <div className="mt-2">
                  <p className="mb-1 text-sm font-medium">
                    {t("documents.filesSelected", { count: batchFilesToUpload.length })}
                  </p>
                  <ScrollArea className="h-24 w-full rounded-md border p-2">
                    <ul className="text-xs">
                      {Array.from(batchFilesToUpload).map((file, index) => (
                        <li key={index} className="border-b border-gray-100 py-1 last:border-0">
                          {file.name} ({(file.size / 1024).toFixed(1)} KB)
                        </li>
                      ))}
                    </ul>
                  </ScrollArea>
                </div>
              )}
            </div>
          ) : (
            <div>
              <Label htmlFor="text" className="mb-2 block">
                {t("documents.text")}
              </Label>
              <Textarea
                id="text"
                value={textContent}
                onChange={e => setTextContent(e.target.value)}
                placeholder={t("documents.enterTextContent")}
                rows={6}
              />
            </div>
          )}

          <div>
            <Label htmlFor="metadata" className="mb-2 block">
              {t("documents.metadata")}
            </Label>
            <Textarea
              id="metadata"
              value={metadata}
              onChange={e => setMetadata(e.target.value)}
              placeholder='{"key": "value"}'
              rows={3}
            />
          </div>

          <div>
            <Label htmlFor="rules" className="mb-2 block">
              {t("documents.rules")}
            </Label>
            <Textarea
              id="rules"
              value={rules}
              onChange={e => setRules(e.target.value)}
              placeholder='[{"type": "metadata_extraction", "schema": {...}}]'
              rows={3}
            />
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="useColpali"
              checked={useColpali}
              onCheckedChange={checked => setUseColpali(checked === true)}
            />
            <Label
              htmlFor="useColpali"
              className="cursor-pointer text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              onClick={() => setUseColpali(!useColpali)}
            >
              {t("documents.useColpali")}
            </Label>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setShowUploadDialog(false)}>
            {t("common.cancel")}
          </Button>
          <Button
            onClick={() => {
              if (uploadType === "file") {
                onFileUpload(fileToUpload, metadata, rules, useColpali);
              } else if (uploadType === "batch") {
                onBatchFileUpload(batchFilesToUpload, metadata, rules, useColpali);
              } else {
                onTextUpload(textContent, metadata, rules, useColpali);
              }
            }}
            disabled={loading}
          >
            {loading ? t("documents.uploading") : t("common.upload")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export { UploadDialog, type UploadDialogProps };

// Export these values as a custom hook for easy access from the parent component
// Custom hook to provide upload dialog state management functionality
export const useUploadDialog = () => {
  // Define all state variables needed for the upload process
  const [uploadType, setUploadType] = useState<"file" | "text" | "batch">("file");
  const [textContent, setTextContent] = useState("");
  // This state is used by the parent component during file upload process
  const [fileToUpload, setFileToUpload] = useState<File | null>(null);
  const [batchFilesToUpload, setBatchFilesToUpload] = useState<File[]>([]);
  const [metadata, setMetadata] = useState("{}");
  const [rules, setRules] = useState("[]");
  const [useColpali, setUseColpali] = useState(true);
  const [showUploadDialog, setShowUploadDialog] = useState(false);

  const resetUploadDialog = () => {
    setUploadType("file");
    setFileToUpload(null);
    setBatchFilesToUpload([]);
    setTextContent("");
    setMetadata("{}");
    setRules("[]");
    setUseColpali(true);
  };

  return {
    uploadType,
    setUploadType,
    textContent,
    setTextContent,
    fileToUpload,
    setFileToUpload,
    batchFilesToUpload,
    setBatchFilesToUpload,
    metadata,
    setMetadata,
    rules,
    setRules,
    useColpali,
    setUseColpali,
    showUploadDialog,
    setShowUploadDialog,
    resetUploadDialog,
  };
};
