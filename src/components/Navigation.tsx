import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Music, BookOpen, Heart, Crown, Home, Box, MapPin, Sparkles, Glasses } from "lucide-react";

const Navigation = () => {
  const location = useLocation();

  const navItems = [
    { path: "/", label: "Home", icon: Home },
    { path: "/travel-logs", label: "Travel Logs", icon: MapPin },
    { path: "/ai-composer", label: "AI Composer", icon: Sparkles },
    { path: "/vr-experience", label: "VR Experience", icon: Glasses },
    { path: "/education", label: "Education", icon: BookOpen },
    { path: "/wellness", label: "Wellness", icon: Heart },
    { path: "/vr-ar", label: "VR/AR", icon: Box },
    { path: "/premium", label: "Premium", icon: Crown },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-lg border-b border-border">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-2 font-bold text-xl">
            <Music className="w-6 h-6 text-primary" />
            <span>Melodic Airways Transforming Flight</span>
          </Link>

          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Button
                  key={item.path}
                  variant={isActive ? "default" : "ghost"}
                  asChild
                  size="sm"
                >
                  <Link to={item.path} className="flex items-center gap-2">
                    <Icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                </Button>
              );
            })}
          </div>

   
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
