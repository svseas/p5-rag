import createMiddleware from "next-intl/middleware";
import { locales } from "./i18n";

export default createMiddleware({
  // A list of all locales that are supported
  locales,

  // Used when no locale matches
  defaultLocale: "vi",

  // Don't use a locale prefix for the default locale
  localePrefix: "as-needed",
});

export const config = {
  // Match all pathnames except for API routes, static files, and Next.js internals
  matcher: ["/((?!api|_next|_vercel|.*\\..*).*)"],
};
