import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navigation from "./components/Navigation";
import Index from "./pages/Index";
import Education from "./pages/Education";
import Wellness from "./pages/Wellness";
import Premium from "./pages/Premium";
import VRAR from "./pages/VRAR";
import TravelLogs from "./pages/TravelLogs";
import AIGenreComposer from "./pages/AIGenreComposer";
import VRExperience from "./pages/VRExperience";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Navigation />
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/education" element={<Education />} />
          <Route path="/wellness" element={<Wellness />} />
          <Route path="/premium" element={<Premium />} />
          <Route path="/vr-ar" element={<VRAR />} />
          <Route path="/travel-logs" element={<TravelLogs />} />
          <Route path="/ai-composer" element={<AIGenreComposer />} />
          <Route path="/vr-experience" element={<VRExperience />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
