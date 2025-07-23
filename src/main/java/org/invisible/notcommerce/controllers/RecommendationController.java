package org.invisible.notcommerce.controllers;

import org.invisible.notcommerce.services.RecommendationEngine;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("api/recommendation")
public class RecommendationController {
   private final RecommendationEngine recommendationEngine;

   public RecommendationController(RecommendationEngine recommendationEngine) {
        this.recommendationEngine = recommendationEngine;
    }

   @GetMapping
   public String recommendation() {
        recommendationEngine.trainModel();
        return "Recommendation successful";
   }
}
