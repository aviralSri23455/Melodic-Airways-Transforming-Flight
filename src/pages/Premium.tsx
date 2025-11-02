import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Check, Crown, Download, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const Premium = () => {
  const plans = [
    {
      name: "Free",
      price: "$0",
      period: "forever",
      features: [
        "Basic route compositions",
        "Standard MIDI export",
        "Community access",
        "5 compositions per month",
      ],
      cta: "Current Plan",
      variant: "outline" as const,
    },
    {
      name: "Premium",
      price: "$9.99",
      period: "per month",
      features: [
        "Unlimited compositions",
        "High-res audio exports (WAV, FLAC)",
        "Advanced AI genre models",
        "Priority rendering",
        "Custom soundfonts",
        "Collaboration features",
        "Analytics dashboard",
      ],
      cta: "Upgrade Now",
      variant: "default" as const,
      popular: true,
    },
    {
      name: "Enterprise",
      price: "Custom",
      period: "contact us",
      features: [
        "Everything in Premium",
        "API access",
        "White-label solution",
        "Custom integrations",
        "Dedicated support",
        "Educational licenses",
        "Commercial usage rights",
      ],
      cta: "Contact Sales",
      variant: "outline" as const,
    },
  ];

  const premiumFeatures = [
    {
      icon: Download,
      title: "High-Resolution Exports",
      description: "Export your compositions in WAV, FLAC, and other professional formats",
    },
    {
      icon: Sparkles,
      title: "AI Genre Models",
      description: "Access advanced PyTorch models for genre-specific compositions",
    },
    {
      icon: Crown,
      title: "Priority Processing",
      description: "Get your compositions generated faster with priority queue access",
    },
  ];

  return (
    <div className="min-h-screen py-24 px-4">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-12">
          <Crown className="w-16 h-16 mx-auto mb-4 text-primary" />
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Premium Features
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Unlock the full potential of FlightSymphony with premium features
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-3 mb-16">
          {plans.map((plan) => (
            <Card 
              key={plan.name}
              className={`relative ${plan.popular ? 'ring-2 ring-primary shadow-lg' : ''}`}
            >
              {plan.popular && (
                <Badge className="absolute -top-3 left-1/2 -translate-x-1/2">
                  Most Popular
                </Badge>
              )}
              <CardHeader>
                <CardTitle className="text-2xl">{plan.name}</CardTitle>
                <CardDescription>
                  <span className="text-3xl font-bold text-foreground">{plan.price}</span>
                  <span className="text-muted-foreground"> / {plan.period}</span>
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <ul className="space-y-3">
                  {plan.features.map((feature) => (
                    <li key={feature} className="flex items-start gap-2">
                      <Check className="w-5 h-5 text-primary shrink-0 mt-0.5" />
                      <span className="text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button 
                  variant={plan.variant}
                  className="w-full"
                  size="lg"
                >
                  {plan.cta}
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="space-y-8">
          <h2 className="text-3xl font-bold text-center">Premium Features in Detail</h2>
          <div className="grid gap-6 md:grid-cols-3">
            {premiumFeatures.map((feature) => {
              const Icon = feature.icon;
              return (
                <Card key={feature.title}>
                  <CardHeader>
                    <Icon className="w-12 h-12 mb-4 text-primary" />
                    <CardTitle>{feature.title}</CardTitle>
                    <CardDescription>{feature.description}</CardDescription>
                  </CardHeader>
                </Card>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Premium;
